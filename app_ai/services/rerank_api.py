# rerank_api.py

import requests


def openai_rerank(api_key, base_url, model, query, documents):
    """
    兼容 OpenAI / 类OpenAI rerank接口（如 Cohere风格）
    """
    url = f"{base_url.rstrip('/')}/rerank"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "query": query,
        "documents": documents
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)

    if resp.status_code != 200:
        raise Exception(f"Rerank请求失败: {resp.text}")

    data = resp.json()

    # 标准返回：[{index: 0, relevance_score: 0.9}, ...]
    return data["results"]


def call_rerank(provider, api_key, base_url, model, query, documents):
    provider = (provider or "").lower()

    # 当前大部分平台兼容 OpenAI/Cohere 格式
    return openai_rerank(api_key, base_url, model, query, documents)
