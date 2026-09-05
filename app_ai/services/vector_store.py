# coding:utf-8
from loguru import logger
from django.db.models import Q
from app_ai.models import DocAISection
import numpy as np

# 向量归一化处理
def normalize(vec):
    vec = np.array(vec)
    norm = np.linalg.norm(vec)
    # 如果模长为 0，直接返回原向量，避免除以 0
    if norm == 0:
        return vec
    return vec / norm


def normalize_list(vec):
    return normalize(vec).tolist()


# VectorStore 抽象层
class BaseVectorStore:

    def add(self, item):
        raise NotImplementedError

    def batch_add(self, items: list):
        """items = [{id, vector, metadata}]"""
        for item in items:
            self.add(**item)

    def search(self, query_vector, top_k=20, **kwargs):
        raise NotImplementedError

    def delete_by_doc(self, doc_id: int):
        raise NotImplementedError

    def delete_by_section(self, section_id: int):
        raise NotImplementedError

    def delete_by_project(self, project_id: int):
        raise NotImplementedError


# 向量存储的数据库实现
class DBVectorStore(BaseVectorStore):

    def serialize_section(self, section):
        return {
            "id": section.id,
            "doc_id": section.doc_id,
            "doc_title": section.doc_title,
            "section_title": section.section_title,
            "title_path": section.title_path,
            "content": section.content,
            "order": section.order,
        }

    def search(self, query_vector, top_k=20, scope=None, threshold=0.2):

        # 仅已发布文档的切片参与检索；scope 再按用户权限过滤
        sections = DocAISection.objects.exclude(embedding=None).filter(doc__status=1)
        if scope:
            sections = sections.filter(
                Q(doc__top_doc__in=scope["project_ids"]) |
                Q(doc_id__in=scope["doc_ids"])
            )

        # 基础边界检查
        if query_vector is None or top_k <= 0:
            return []

        try:
            # 1. 向量化准备
            q_vec = normalize(query_vector)
            dim = q_vec.shape[0]

            # 2. 预过滤：只取 id + embedding 两列，避免把全文 content 全部读进内存
            valid_data = [
                (sec_id, emb)
                for sec_id, emb in sections.values_list("id", "embedding")
                if emb is not None and len(emb) == dim
            ]

            if not valid_data:
                return []

            # 3. 矩阵运算
            ids, embeddings = zip(*valid_data)
            scores = np.dot(np.array(embeddings), q_vec)

            # 4. 阈值过滤 (Boolean Masking)
            mask = scores >= threshold
            if not np.any(mask):
                return []

            # 应用掩码获取及格的分数和对应的索引
            filtered_scores = scores[mask]
            passing_ids = np.array(ids)[np.where(mask)[0]]

            # 5. Top-K 选取
            k = min(top_k, len(passing_ids))
            # 在过滤后的集合中找 Top-K
            rel_indices = np.argpartition(filtered_scores, -k)[-k:]
            # 按分数从高到低排序
            rel_indices = rel_indices[np.argsort(filtered_scores[rel_indices])[::-1]]

            top_ids = [int(passing_ids[i]) for i in rel_indices]
            top_scores = [float(filtered_scores[i]) for i in rel_indices]

            # 6. 只对命中的 top-k 回查完整字段
            section_map = {
                sec.id: sec
                for sec in DocAISection.objects.filter(id__in=top_ids)
            }

            # 7. 构造最终返回列表
            return [
                {
                    "section": self.serialize_section(section_map[sec_id]),
                    "score": round(score, 4)
                }
                for sec_id, score in zip(top_ids, top_scores)
            ]

        except Exception as e:
            logger.error(f"DBVectorStore search error: {e}")
            return []

    def add(self, item):
        from app_ai.services.ai_router import get_embedding
        try:
            embedding = get_embedding(item.embedding_text)
            item.embedding = normalize_list(embedding)  # 对向量进行归一化处理
            item.embedding_status = "done"
            item.save(update_fields=["embedding", "embedding_status"])
        except Exception as e:
            item.embedding_status = "failed"
            item.save(update_fields=["embedding_status"])
            logger.error(f"[embedding失败] section={item.id} {e}")
            raise e

    def delete_by_doc(self, doc_id):
        # DB模式不需要
        pass

    def delete_by_section(self, section_id: int):
        pass

    def delete_by_project(self, project_id: int):
        # DB模式下切片行删除即向量删除，无需额外处理
        pass


# 向量入口（开源版仅支持数据库存储）
def get_vector_store():
    return DBVectorStore()


# 向量检索
def vector_search(query, scope=None):
    from app_ai.services.ai_router import get_embedding

    store = DBVectorStore()
    query_vec = normalize(get_embedding(query))

    results = store.search(query_vec, scope=scope)

    return results
