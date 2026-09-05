# coding:utf-8
from loguru import logger
from app_admin.models import SysSetting


# 返回文档的文本内容
def get_doc_content(doc):
    if doc.editor_mode == 3:
        return doc.content
    return doc.pre_content


# 获取系统配置
def get_sys_value(types, name, default=None, *, allow_empty=False):
    obj = SysSetting.objects.filter(types=types, name=name).first()
    if not obj:
        return default
    if not allow_empty and not obj.value:
        return default
    return obj.value


# 同步文档内容到AI知识库（内置引擎：全站已发布文档切片索引，同步执行）
def ai_sync_doc(doc):
    try:
        # 获取AI状态
        ai_status = getattr(SysSetting.objects.filter(types='ai', name='ai_status').first(), 'value', '')
        if ai_status != '1':
            return False, ''
        # 仅已发布文档参与切片索引
        if doc.status != 1:
            return False, ''
        from app_ai.util.llm_wiki import build_sections_pipeline
        build_sections_pipeline(doc)
        return True, ''
    except Exception as e:
        logger.exception("同步AI知识库文档异常：" + repr(e))
        return False, ''


# 删除AI知识库文档切片
def ai_del_doc(doc_id):
    try:
        from app_ai.models import DocAISection
        from app_ai.services.bm25_store import invalidate_bm25_cache
        DocAISection.objects.filter(doc_id=doc_id).delete()
        invalidate_bm25_cache()
        return True, ''
    except Exception as e:
        logger.error("删除AI知识库文档切片异常：" + repr(e))
        return False, ''
