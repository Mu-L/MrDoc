# coding:utf-8

from openai import OpenAI
import requests


def openai_embedding(api_key, base_url, model, text):
    client = OpenAI(api_key=api_key, base_url=base_url)

    resp = client.embeddings.create(
        model=model,
        input=text[:2000]  # 防止超长
    )

    return resp.data[0].embedding


def openai_like_embedding(api_key, base_url, model, text):
    url = f"{base_url.rstrip('/')}/embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": text[:2000]
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)

    if resp.status_code != 200:
        raise Exception(f"Embedding请求失败: {resp.text}")

    data = resp.json()

    return data["data"][0]["embedding"]


def call_embedding(provider, api_key, base_url, model, text):
    provider = (provider or "").lower()

    if provider == "openai":
        return openai_embedding(api_key, base_url, model, text)

    elif provider == "siliconflow":
        return openai_like_embedding(api_key, base_url, model, text)

    elif provider == "giteeai":
        return openai_embedding(api_key, base_url, model, text)

    else:
        return openai_like_embedding(api_key, base_url, model, text)
