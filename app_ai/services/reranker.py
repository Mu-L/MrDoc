# coding:utf-8
import logging
import jieba
from typing import List, Dict, Any, Optional
from app_admin.utils import decrypt_data
from app_ai.models import AIProvider
from app_ai.services.rerank_api import call_rerank

logger = logging.getLogger(__name__)


def _get_section_field(sec, name, default=""):
    """
    兼容 dict（DB 序列化返回）与 Django 模型两种 section 表示，
    读取字段值。dict 无法用 getattr 取字段，否则会静默返回默认值导致重排失效。
    """
    if isinstance(sec, dict):
        return sec.get(name, default) or default
    return getattr(sec, name, default) or default


class BaseReranker:
    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError


class SimpleReranker(BaseReranker):
    MIN_TERM_LEN = 2

    STOP_WORDS = {
        "如何",
        "怎么",
        "请问",
        "是否",
        "可以",
        "一个",
        "一下",
        "什么",
        "为什么",
        "怎样",
        "进行",
        "使用",
        "有关",
        "关于",
        "以及",
        "或者",
        "里面",
        "这里",
        "那个",
        "这个",
    }

    def _tokenize(self, text: str) -> set:
        """
        中文分词 + 去停用词
        """
        return {
            word.strip().lower()
            for word in jieba.cut(text)
            if len(word.strip()) >= self.MIN_TERM_LEN
               and word.strip() not in self.STOP_WORDS
        }

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not results:
            return []

        query_terms = self._tokenize(query)

        if not query_terms:
            return results[:top_k]

        for item in results:
            sec = item["section"]

            base_score = float(item.get("score", 0))

            doc_title = _get_section_field(sec, "doc_title")
            section_title = _get_section_field(sec, "section_title")
            content = _get_section_field(sec, "content")

            title_text = (
                f"{doc_title} {section_title}"
            ).lower()

            content_text = content.lower()

            # 标题命中
            title_hits = sum(
                1
                for term in query_terms
                if term in title_text
            )

            # 内容命中
            content_hits = sum(
                1
                for term in query_terms
                if term in content_text
            )

            total_terms = len(query_terms)

            title_ratio = title_hits / total_terms
            content_ratio = content_hits / total_terms

            # 标题权重大于内容
            keyword_score = (
                    title_ratio * 0.7 +
                    content_ratio * 0.3
            )

            # Embedding 主导
            final_score = (
                    base_score * 0.9 +
                    keyword_score * 0.1
            )

            item["rerank_score"] = round(keyword_score, 4)
            item["final_score"] = round(final_score, 6)

            # 方便调试
            item["match_detail"] = {
                "title_hits": title_hits,
                "content_hits": content_hits,
                "query_terms": len(query_terms),
                "base_score": round(base_score, 4),
                "keyword_score": round(keyword_score, 4),
            }

        results.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        return results[:top_k]


class RemoteModelReranker(BaseReranker):
    def __init__(self, provider: str, api_key: str, base_url: str, model: str, max_content_length: int = 500):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_content_length = max_content_length

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not results:
            return []

        # 构造 documents
        docs = []
        for item in results:
            sec = item["section"]
            title = _get_section_field(sec, 'section_title')
            content = _get_section_field(sec, 'content')
            text = f"{title}\n{content[:self.max_content_length]}".strip()
            docs.append(text)

        try:
            # 调用远程 rerank
            rerank_results = call_rerank(
                self.provider,
                self.api_key,
                self.base_url,
                self.model,
                query,
                docs
            )

            # 写回分数
            for r in rerank_results:
                idx = r.get("index")
                # 边界检查
                if idx is None or not isinstance(idx, int) or idx >= len(results) or idx < 0:
                    logger.warning(f"Invalid rerank index: {idx}")
                    continue

                score = r.get("relevance_score", 0)
                results[idx]["rerank_score"] = score
                results[idx]["final_score"] = results[idx].get("score", 0) + score

            # 排序
            results.sort(key=lambda x: x["final_score"], reverse=True)
            return results[:top_k]

        except TimeoutError as e:
            logger.warning(f"Rerank timeout: {e}")
            return results[:top_k]
        except Exception as e:
            logger.exception(f"Remote rerank error: {e}")
            return results[:top_k]


def get_reranker() -> Optional[BaseReranker]:
    """获取重排器实例。

    仅当配置了可用的远程 reranker（AIProvider model_type="reranker"）时
    返回 RemoteModelReranker；否则返回 None，由调用方决定跳过重排。
    （未配置远程重排时直接按 RRF 融合序输出更优）
    """
    try:
        ai_model = AIProvider.objects.filter(
            is_active=True,
            model_type="reranker"
        ).order_by('-updated_at').first()

        if ai_model:
            try:
                api_key = decrypt_data(ai_model.api_key)
            except Exception as e:
                logger.error(f"Failed to decrypt API key: {e}")
                return None

            return RemoteModelReranker(
                provider=ai_model.provider,
                api_key=api_key,
                base_url=ai_model.base_url or '',
                model=ai_model.model_name,
            )

        return None

    except Exception as e:
        logger.exception(f"Error initializing reranker: {e}")
        return None
