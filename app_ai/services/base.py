# coding:utf-8

import json
import requests


def sse_yield(event: str, data):
    """统一的SSE事件格式"""
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, ensure_ascii=False)
    else:
        data_str = str(data)
    return f"data: {json.dumps({'event': event, 'data': data_str})}\n\n"


def stream_sse_request(url, headers, body):
    """通用流式请求封装"""
    with requests.post(url, headers=headers, json=body, stream=True, timeout=180) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf-8")


def parse_sse_chunk(chunk):
    """
    把各种provider返回的chunk统一解析成结构化数据
    """
    try:
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk

        if not text.startswith("data:"):
            return None

        data_str = text[5:].strip()

        if data_str == "[DONE]":
            return {"event": "message_end"}

        data = json.loads(data_str)

        return data

    except Exception:
        return None
