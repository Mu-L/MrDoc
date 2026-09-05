# coding:utf-8
# hybrid_search.py

# RRF 常数：越大，排名差异对融合分的影响越平缓（业界常用值）
RRF_K = 60


def _section_id(section):
    """
    兼容 dict（DB 序列化）与 Django 模型两种表示，
    取切片主键作为去重 key。
    """
    if isinstance(section, dict):
        return section.get("id", section.get("section_id"))
    return getattr(section, "id", None)


def hybrid_merge(vector_results, bm25_results, k=RRF_K, vector_weight=0.6, bm25_weight=0.4):
    """
    RRF（Reciprocal Rank Fusion）融合：

        score = Σ weight_i * k / (k + rank_i)

    只依赖各路排名、不依赖原始分数量纲，天然规避向量余弦分（0~1）
    与 BM25 分（无上界）之间的归一化失真问题；
    k/(k+rank) 变体把融合分映射到 0~1，与下游来源过滤阈值兼容。

    权重沿用原 alpha=0.6 的语义：向量（语义）召回为主，BM25（关键词）为辅。
    """
    merged = {}

    def _ensure(key, section):
        return merged.setdefault(key, {
            "section": section,
            "vector_rank": None,
            "bm25_rank": None,
            "vector_score": 0,
            "bm25_score": 0,
        })

    for rank, item in enumerate(vector_results, start=1):
        sec = item["section"]
        key = _section_id(sec)
        if key is None:
            continue
        entry = _ensure(key, sec)
        if entry["vector_rank"] is None:
            entry["vector_rank"] = rank
            entry["vector_score"] = item.get("score", 0)

    for rank, item in enumerate(bm25_results, start=1):
        sec = item["section"]
        key = _section_id(sec)
        if key is None:
            continue
        entry = _ensure(key, sec)
        if entry["bm25_rank"] is None:
            entry["bm25_rank"] = rank
            entry["bm25_score"] = item.get("score", 0)

    results = []
    for entry in merged.values():
        score = 0
        if entry["vector_rank"] is not None:
            score += vector_weight * k / (k + entry["vector_rank"])
        if entry["bm25_rank"] is not None:
            score += bm25_weight * k / (k + entry["bm25_rank"])

        results.append({
            "section": entry["section"],
            "score": round(score, 4),
            "vector_score": round(entry["vector_score"], 4),
            "bm25_score": round(entry["bm25_score"], 4),
            "vector_rank": entry["vector_rank"],
            "bm25_rank": entry["bm25_rank"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
