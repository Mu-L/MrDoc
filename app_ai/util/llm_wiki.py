# coding:utf-8

from loguru import logger
from app_ai.models import DocAISection
from app_ai.services.bm25_store import bm25_search
from app_ai.services.hybrid_search import hybrid_merge
from app_ai.services.reranker import get_reranker
from app_ai.services.vector_store import vector_search, get_vector_store
from app_ai.util.prompts import build_answer_prompt
from app_ai.services.base import parse_sse_chunk
from app_ai.services.ai_router import call_llm_core
from app_ai.util.spliter.markdown_chunk_spliter import chunk_markdown_document

from app_doc.models import DocTag
from app_doc.search_utils import get_user_access_scope


# 模型调用 - 同步
def call_llm(prompt: str, request=None) -> str:
    result = call_llm_core(prompt, request, mode="sync")
    # result 是 dict
    if isinstance(result, dict):
        return result.get("content", "").strip()

    # 兜底（兼容旧逻辑）
    full_text = ""
    for chunk in result:
        data = parse_sse_chunk(chunk)
        if data and data.get("event") == "message":
            full_text += data.get("answer", "")
    return full_text.strip()

# 模型调用 - 流式
def call_llm_stream(prompt: str, request=None):
    result = call_llm_core(prompt, request, mode="stream")

    # 必须直接 yield，不要拼接
    for chunk in result:
        yield chunk

# 构建切片和向量索引（同步执行）
def build_sections_pipeline(doc):
    from app_ai.utils import get_doc_content

    content = get_doc_content(doc)
    doc_tags = ",".join(
        DocTag.objects.filter(doc_id=doc.id)
        .select_related("tag")
        .values_list("tag__name", flat=True)
    )
    # 1.切分
    chunks = chunk_markdown_document(content, doc.name, doc_tags)

    # 切分结果为空时绝不走后面的软删除/清向量，否则一次异常就会把该文档索引清空
    if not chunks:
        logger.warning(f"[AI索引] 文档切片为空，保留原有索引: doc_id={doc.id}")
        return {
            "doc_id": doc.id,
            "total": 0,
            "failed": 0,
            "skipped": True
        }

    # 2.先“软删除”（避免中断导致数据丢失）
    DocAISection.objects.filter(doc=doc).update(source_type="deprecated")

    new_sections = []

    # 3.先创建所有 section（状态=待处理）
    for i, chunk in enumerate(chunks):
        section = DocAISection.objects.create(
            doc=doc,
            doc_title=doc.name,
            title_path=chunk['title_path'],
            section_title=chunk['section_title'],
            content=chunk['content'],
            embedding_text=chunk['embedding_text'],
            order=i,
            source_type="raw",
            embedding_status="pending",
            llm_status="pending"
        )
        new_sections.append(section)

    # 4.逐个处理（embedding）
    vector_store = get_vector_store()
    vector_store.delete_by_doc(doc.id)  # 清理旧的向量
    failed_count = 0
    for section in new_sections:
        # ===== embedding =====
        try:
            vector_store.add(section)
        except Exception as e:
            logger.error(f"[embedding失败] section={section.id} {e}")
            failed_count += 1

    # 5.清理旧数据（最后再删）
    DocAISection.objects.filter(doc=doc, source_type="deprecated").delete()
    return {
        "doc_id": doc.id,
        "total": len(new_sections),
        "failed": failed_count
    }

# 重建section索引
def rebuild_section(section):
    # ===== embedding =====
    vector_store = get_vector_store()
    vector_store.add(section)


# 向量 + BM25混合检索（RRF融合）
def hybrid_search(query, scope=None):
    # 1. 向量召回（内部完成权限过滤、相似度阈值与 top_k 截断）；
    #    embedding 服务异常时降级为纯 BM25 检索，保证链路可用
    try:
        vector_results = vector_search(query, scope)
    except Exception as e:
        logger.warning(f"[向量召回失败，降级为纯BM25检索] {e}")
        vector_results = []

    # 2. BM25召回（同一权限范围，补齐关键词精确命中的盲区）
    try:
        bm25_results = bm25_search(query, scope, top_k=20)
    except Exception as e:
        logger.warning(f"[BM25召回失败，降级为纯向量检索] {e}")
        bm25_results = []

    # 3. RRF融合
    return hybrid_merge(vector_results, bm25_results)


# RAG搜索
def rag_search(question, user, project_ids=None):
    # 筛选用户有权限的文集ID列表和文档ID列表
    scope = get_user_access_scope(user)
    if project_ids:
        scope["project_ids"] = [pid for pid in scope["project_ids"] if pid in project_ids]

    # 混合检索召回（向量 + BM25 经 RRF 融合；BM25 异常时自动降级为纯向量结果）
    results = hybrid_search(question, scope)

    # 重排：仅当配置了远程 reranker 时启用；未配置时按 RRF 融合序直接截断
    reranker = get_reranker()
    if reranker is not None:
        results = reranker.rerank(question, results, top_k=8)
    else:
        results = results[:8]

    if results:
        return {
            "type": "rag",
            "sections": [
                {
                    "content": item['section']['content'],
                    "doc_title": item['section']['doc_title'],
                    "section_title": item['section']['section_title'],
                    "title_path": item['section']['title_path'],
                    "doc_id": item['section']['doc_id'],
                    "score": item['score']
                }
                for item in results
            ]
        }

    # fallback
    return {
        "type": "empty",
        "sections": []
    }


# 生成来源文档
def build_sources(sections):

    doc_map = {}

    for sec in sections:
        doc_id = sec["doc_id"]
        score = sec.get("score", 0)

        # 保留最高分
        if doc_id not in doc_map or score > doc_map[doc_id]["score"]:
            doc_map[doc_id] = sec

    # 按 score 排序
    sorted_secs = sorted(doc_map.values(), key=lambda x: x.get("score", 0), reverse=True)

    return [
        {
            "document_id": sec["doc_id"],
            "title": sec.get("doc_title", ""),
            "score": sec.get("score", 0)
        }
        for sec in sorted_secs
    ]

# 主入口
def ask_ai(question, user, project_ids=None, prompt_id=None):

    answer = ""
    reasoning = ""
    sources = []

    for data in ask_ai_stream(question, user, project_ids, prompt_id):

        event = data.get("event")

        if event == "message":
            answer += data.get("answer", "")

        elif event == "reasoning":
            reasoning += data.get("answer", "")

        elif event == "message_end":
            metadata = data.get("metadata", {})
            sources = metadata.get(
                "retriever_resources",
                []
            )

    return {
        "answer": answer,
        "reasoning": reasoning,
        "sources": sources
    }

# 主入口 - 流式
def ask_ai_stream(question, user, project_ids=None, prompt_id=None):

    # ① 状态：开始检索知识库
    yield {
        "event": "status",
        "status": "searching",
        "message": "知识库检索中..."
    }

    result = rag_search(question, user, project_ids)

    # RAG
    if result["type"] == "rag":
        sections = result["sections"]

        # 文档来源
        # RRF 分数语义：单路第一名分值≈该路权重（向量0.6/BM25 0.4），
        # BM25 单路强命中约 0.37~0.39，阈值需低于该区间才能保留关键词命中的来源
        sources = [
            s for s in build_sources(sections)
            if s["score"] >= 0.35
        ][:3]

        # ② 状态：已检索到相关知识
        yield {
            "event": "status",
            "status": "retrieved",
            "message": f"已检索到 {len(sources)} 篇相关文档"
        }

        # 支持自定义 Prompt
        if prompt_id:
            from app_ai.models import Prompt
            prompt_obj = Prompt.objects.filter(id=prompt_id).first()
            if prompt_obj:
                context = "\n\n".join([
                    f"[{i + 1}]【{sec['section_title']}】\n"
                    f"路径：{sec['title_path']}\n"
                    f"来源：{sec['doc_title']}\n"
                    f"{sec['content']}"
                    for i, sec in enumerate(sections)
                ])
                prompt = prompt_obj.value.replace('{{context}}', context).replace('{{question}}', question)
            else:
                prompt = build_answer_prompt(question, sections)
        else:
            prompt = build_answer_prompt(question, sections)

        # ③ 状态：正在组织回答
        yield {
            "event": "status",
            "status": "answering",
            "message": "正在组织回答..."
        }

        # 直接透传 LLM 流
        for chunk in call_llm_stream(prompt):
            data = parse_sse_chunk(chunk)
            if data and data.get("event") in ["message", "reasoning"]:
                yield data
        yield {
            "event": "message_end",
            "metadata": {
                "retriever_resources": sources
            }
        }
        return

    else:
        # 状态：未检索到相关知识
        yield {
            "event": "status",
            "status": "empty",
            "message": "未检索到相关知识"
        }
        # fallback
        yield {
            "event": "message",
            "answer": "未在知识库中找到相关内容，请尝试更换问题或关键词。"
        }

        yield {
            "event": "message_end",
            "metadata": {
                "retriever_resources": []
            }
        }
