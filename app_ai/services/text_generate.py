# coding:utf-8

from loguru import logger
from openai import OpenAI
import requests
import json


def handle_stream_response_line(line):
    """解析 OpenAI 格式流数据（兼容 reasoning）"""
    if not line.startswith(b"data:"):
        return None

    data_str = line.decode("utf-8")[5:].strip()
    if data_str == "[DONE]":
        return {"event": "message_end"}

    try:
        data = json.loads(data_str)
        delta = data.get("choices", [{}])[0].get("delta", {})
        if "reasoning_content" in delta:
            return {"event": "reasoning", "answer": delta["reasoning_content"]}
        elif "content" in delta:
            return {"event": "message", "answer": delta["content"]}
    except Exception:
        return None


# OpenAI通用文本生成
def call_openai(api_key: str, base_url: str, model: str, prompt: str):
    client = OpenAI(api_key=api_key, base_url=base_url)
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        chunk_json = json.loads(chunk.model_dump_json())

        # reasoning event
        if "event" in chunk_json and chunk_json["event"] == "reasoning":
            yield f"data: {json.dumps({'event': 'reasoning', 'answer': chunk_json['answer']})}\n\n"
        else:
            # 普通输出
            delta = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta:
                yield f"data: {json.dumps({'event': 'message', 'answer': delta})}\n\n"
    yield f"data: {json.dumps({'event': 'message_end'})}\n\n"


def call_openai_sync(api_key, base_url, model, prompt):
    client = OpenAI(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    content = resp.choices[0].message.content

    return {
        "content": content,
        "raw": resp.model_dump()
    }


# GiteeAI 文本生成
def call_giteeai(api_key: str, base_url: str, model: str, prompt: str):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        url = f"{base_url.rstrip('/')}/chat/completions"

        with requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=300
        ) as r:
            r.encoding = 'utf-8'
            r.raise_for_status()
            # 转发数据流
            for chunk in r.iter_lines(decode_unicode=True):
                if not chunk:
                    continue
                if not chunk.startswith("data:"):
                    continue
                data_str = chunk[5:].strip()
                if data_str == "[DONE]":
                    yield f"data: {json.dumps({'event': 'message_end'})}\n\n"
                    break

                try:
                    data = json.loads(data_str)
                except Exception as e:
                    logger.error(f"解析流数据失败: {e}")
                    continue
                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                # 思考内容
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    yield f"data: {json.dumps({'event': 'reasoning', 'answer': reasoning})}\n\n"

                # 普通内容
                content = delta.get("content")
                if content:
                    yield f"data: {json.dumps({'event': 'message', 'answer': content})}\n\n"

            yield f"data: {json.dumps({'event': 'message_end'})}\n\n"

    except Exception as e:
        logger.error(f"请求异常：{repr(e)}")
        yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"


def call_giteeai_sync(api_key, base_url, model, prompt):
    try:
        url = f"{base_url.rstrip('/')}/chat/completions"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60
        )

        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        return {
            "content": content,
            "raw": data
        }
    except Exception as e:
        logger.error(f"[请求LLM失败] {repr(e)}")


# 硅基流动 文本生成
def call_siliconflow(api_key: str, base_url: str, model: str, prompt: str):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        url = f"{base_url.rstrip('/')}/chat/completions"

        with requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=300
        ) as r:
            r.encoding = 'utf-8'
            r.raise_for_status()
            # 转发数据流
            for chunk in r.iter_lines(decode_unicode=True):
                if not chunk:
                    continue
                if not chunk.startswith("data:"):
                    continue
                data_str = chunk[5:].strip()
                if data_str == "[DONE]":
                    yield f"data: {json.dumps({'event': 'message_end'})}\n\n"
                    break

                try:
                    data = json.loads(data_str)
                except Exception as e:
                    logger.error(f"解析流数据失败: {e}")
                    continue
                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                # 思考内容
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    yield f"data: {json.dumps({'event': 'reasoning', 'answer': reasoning})}\n\n"

                # 普通内容
                content = delta.get("content")
                if content:
                    yield f"data: {json.dumps({'event': 'message', 'answer': content})}\n\n"

            yield f"data: {json.dumps({'event': 'message_end'})}\n\n"

    except Exception as e:
        logger.error(f"请求异常：{repr(e)}")
        yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"


def call_siliconflow_sync(api_key, base_url, model, prompt):
    url = f"{base_url.rstrip('/')}/chat/completions"

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=60
    )

    resp.raise_for_status()
    data = resp.json()

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    return {
        "content": content,
        "raw": data
    }


def generic_stream_response(provider: str, api_key: str, base_url: str, model: str, prompt: str):
    """
    使用 requests 手动流式请求（通用实现）。
    适用于：自建代理、非 OpenAI 协议厂商（如 GiteeAI、自定义接口）
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        url = f"{base_url.rstrip('/')}/chat/completions"

        with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as r:
            r.raise_for_status()
            # 转发数据流
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    yield chunk

    except Exception as e:
        logger.error(f"请求异常：{repr(e)}")
        yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"


def generic_sync_response(provider: str, api_key: str, base_url: str, model: str, prompt: str):
    """
    通用非流式请求（适配 OpenAI 协议 / 自建代理）
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        url = f"{base_url.rstrip('/')}/chat/completions"

        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )

        resp.raise_for_status()
        data = resp.json()

        # 兼容 OpenAI 格式
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        return {
            "content": content,
            "raw": data
        }

    except Exception as e:
        logger.error(f"请求异常：{repr(e)}")
        return {
            "content": "",
            "error": str(e)
        }


# 文本生成流式响应
def call_text_generate(provider: str, api_key: str, base_url: str, model: str, prompt: str):
    """
    自动判断使用哪种流式调用方式。
    """
    provider = (provider or "").lower()
    if provider == 'giteeai':
        yield from call_giteeai(api_key, base_url, model, prompt)
    elif provider == 'siliconflow':
        yield from call_siliconflow(api_key, base_url, model, prompt)
    else:
        yield from call_openai(api_key, base_url, model, prompt)


# 文本生成非流式响应
def call_text_generate_sync(provider, api_key, base_url, model, prompt):
    provider = (provider or "").lower()

    if provider == 'giteeai':
        return call_giteeai_sync(api_key, base_url, model, prompt)
    elif provider == 'siliconflow':
        return call_siliconflow_sync(api_key, base_url, model, prompt)
    else:
        return call_openai_sync(api_key, base_url, model, prompt)
