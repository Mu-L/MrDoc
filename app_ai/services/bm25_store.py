# coding:utf-8

import hashlib
import os
import pickle
from collections import OrderedDict
from threading import RLock

import numpy as np
import jieba
from django.db.models import Q
from django.db.models.signals import post_save, post_delete

from app_ai.models import DocAISection
from app_ai.services.rank_bm25 import BM25Okapi

# 分词结果的序列化分隔符（实际语料中不会出现的控制字符，保证 join/split 无损往返）
_TOKEN_SEP = "\x1f"
# 语料低于该规模不落盘分词缓存（构建已足够快，避免小项目写大文件）
_TOKEN_CACHE_MIN_DOCS = 2000
# 分词缓存条目兜底上限：大量删除导致残留膨胀时整体重置
_TOKEN_CACHE_HARD_LIMIT = 200000


def _serialize_section(section):
    """统一序列化为 dict，字段与 DBVectorStore.serialize_section 保持一致"""
    return {
        "id": section.id,
        "doc_id": section.doc_id,
        "doc_title": section.doc_title,
        "section_title": section.section_title,
        "title_path": section.title_path,
        "content": section.content,
        "order": section.order,
    }


class BM25Store:
    """
    BM25 关键词检索器，检索结果与向量召回保持同一结构：
        {"section": {...}, "score": ...}
    """

    def __init__(self, sections):
        """
        sections: DocAISection QuerySet / 模型实例列表

        分词结果按 section 粒度缓存（内存 + 磁盘）：全量分词占索引构建耗时约 96%，
        缓存后仅对新增/变更切片增量分词，重建降至秒级；
        updated_at 未变化的切片直接复用缓存 token，进程重启后同样生效。
        """
        serialized = []
        corpus = []
        cache_changed = False
        new_entries = {}
        token_cache = _load_token_cache()

        for sec in sections:
            serialized.append(_serialize_section(sec))
            updated_at = getattr(sec, "updated_at", None)
            entry = token_cache.get(sec.id)
            if entry is not None and entry[0] == updated_at:
                tokens_joined = entry[1]
            else:
                # 语料包含文档标题与标题路径：用户查询常直接命中文档标题，
                # 仅靠 section_title + content 会丢失这部分关键词召回能力
                text = " ".join(filter(None, [
                    getattr(sec, "doc_title", ""),
                    getattr(sec, "title_path", ""),
                    getattr(sec, "section_title", ""),
                    getattr(sec, "content", ""),
                ]))
                tokens_joined = _TOKEN_SEP.join(jieba.cut(text))
                new_entries[sec.id] = (updated_at, tokens_joined)
                cache_changed = True
            corpus.append(tokens_joined.split(_TOKEN_SEP))

        self.sections = serialized
        # 空语料下 BM25Okapi 会除零，置空后检索直接返回 []
        self.bm25 = BM25Okapi(corpus) if corpus else None

        if cache_changed:
            _save_token_cache(new_entries, len(corpus))

    def search(self, query, top_k=20, min_score=0.5):
        """
        min_score: BM25 分数下限，低于该值视为无有效关键词命中，
        不进入融合，避免无关节点污染 RRF 排名。
        """
        if self.bm25 is None:
            return []

        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)

        idxs = np.argsort(scores)[-top_k:][::-1]

        results = []
        for i in idxs:
            score = float(scores[i])
            if score <= min_score:
                continue
            results.append({
                "section": self.sections[int(i)],
                "score": score,
                "source": "bm25"
            })
        return results


# ===== 分词结果磁盘缓存 =====
# jieba 全量分词是索引构建的主要开销，按 section 粒度持久化到 config/bm25_token_cache.pkl，
# 条目为 {section_id: (updated_at, 分词结果)}，updated_at 变化即重新分词；
# 进程重启后复用，数据变更后仅对变更切片增量分词。
_token_cache = None
_token_cache_loaded = False
_token_cache_lock = RLock()


def _token_cache_path():
    from django.conf import settings
    return os.path.join(settings.BASE_DIR, "config", "bm25_token_cache.pkl")


def _load_token_cache():
    global _token_cache, _token_cache_loaded
    if _token_cache_loaded:
        return _token_cache
    with _token_cache_lock:
        if not _token_cache_loaded:
            try:
                with open(_token_cache_path(), "rb") as f:
                    _token_cache = pickle.load(f)
                if not isinstance(_token_cache, dict):
                    _token_cache = {}
            except Exception:
                _token_cache = {}
            _token_cache_loaded = True
    return _token_cache


def _save_token_cache(entries, corpus_size):
    """合并增量分词结果并原子落盘；失败不影响检索，仅下次重建时重新分词"""
    global _token_cache
    with _token_cache_lock:
        cache = _load_token_cache()
        cache.update(entries)
        if len(cache) > _TOKEN_CACHE_HARD_LIMIT:
            # 兜底：大量删除残留导致膨胀时整体重置，代价是下次全量重建
            cache = dict(entries)
            _token_cache = cache
        if corpus_size < _TOKEN_CACHE_MIN_DOCS:
            return  # 小语料构建很快，不值得落盘
        path = _token_cache_path()
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ===== BM25 索引缓存 =====
# 语料需全量 jieba 分词，构建成本高，按 (数据版本, scope签名) 缓存复用；
# DocAISection 的语料字段增删改时通过信号自增版本号，缓存自动失效。
_BM25_CACHE = OrderedDict()   # key -> BM25Store
_BM25_CACHE_MAX = 8
_BM25_CACHE_LOCK = RLock()
_index_version = 0

# 仅这些字段影响 BM25 语料，其余字段（embedding_status 等）的保存不触发索引失效
_CORPUS_FIELDS = {"content", "section_title", "doc_title", "title_path"}


def _bump_index_version(sender, instance, **kwargs):
    global _index_version
    update_fields = kwargs.get("update_fields")
    if update_fields is not None and not (_CORPUS_FIELDS & set(update_fields)):
        return
    _index_version += 1


# QuerySet.update() 不触发信号（如索引 pipeline 的软删除 update），
# 后续的 .delete() 会触发；确有需要可手动调 invalidate_bm25_cache()
post_save.connect(_bump_index_version, sender=DocAISection)
post_delete.connect(_bump_index_version, sender=DocAISection)


def invalidate_bm25_cache():
    """手动失效所有 BM25 索引缓存（用于不触发信号的批量操作之后）"""
    global _index_version
    with _BM25_CACHE_LOCK:
        _BM25_CACHE.clear()
        _index_version += 1


def _scope_key(scope):
    if not scope:
        return "all"
    raw = "p:{}|d:{}".format(
        ",".join(str(i) for i in sorted(scope.get("project_ids") or [])),
        ",".join(str(i) for i in sorted(scope.get("doc_ids") or [])),
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _corpus_queryset(scope):
    """
    仅已发布文档的切片参与检索，并与向量召回同一权限语义
    （doc__top_doc 或 doc_id 命中 scope），排除已废弃切片；
    .only 避免把大字段 embedding 读进内存。
    """
    qs = DocAISection.objects.exclude(source_type="deprecated").filter(
        doc__status=1
    ).only(
        "id", "doc_id", "doc_title", "section_title", "title_path", "content", "order", "updated_at"
    )
    if scope:
        qs = qs.filter(
            Q(doc__top_doc__in=scope["project_ids"]) |
            Q(doc_id__in=scope["doc_ids"])
        )
    return qs


def get_bm25_store(scope=None):
    key = (_index_version, _scope_key(scope))

    with _BM25_CACHE_LOCK:
        store = _BM25_CACHE.get(key)
        if store is not None:
            _BM25_CACHE.move_to_end(key)
            return store
        while len(_BM25_CACHE) >= _BM25_CACHE_MAX:
            _BM25_CACHE.popitem(last=False)

    # 构建过程不放锁内，避免阻塞其他线程读缓存
    store = BM25Store(_corpus_queryset(scope))

    with _BM25_CACHE_LOCK:
        # 构建期间数据已变更则不缓存本次结果，下次重建
        if key[0] == _index_version:
            _BM25_CACHE[key] = store
    return store


def bm25_search(query, scope=None, top_k=20):
    return get_bm25_store(scope).search(query, top_k=top_k)
