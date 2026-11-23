# -*- coding: utf-8 -*-
r"""
phase3_embedding.py —— 第三步：语义检索（embedding + 余弦相似度）

Phase 2 用"字面重叠"检索（多少钱 vs 月费299元 → 对不上）；
Phase 3 把文字变成向量，用"余弦相似度"衡量语义相关，解决这个问题。

流程不变，只换 retrieve：切块 → embed 成向量 → 余弦相似度检索 → 生成

运行： .\.venv\Scripts\python.exe phase3_embedding.py
首次运行会自动从 hf-mirror.com 下载 bge-small-zh-v1.5（约 95MB），耐心等。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # HuggingFace 国内镜像

import requests
from sentence_transformers import SentenceTransformer
from config import DEEPSEEK_API_KEY

URL = "https://api.deepseek.com/chat/completions"


def chat(messages):
    r = requests.post(
        URL,
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": "deepseek-chat", "messages": messages,
              "temperature": 0.3, "max_tokens": 500},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def load_and_chunk(path):
    text = open(path, encoding="utf-8").read()
    return [p.strip() for p in text.split("\n\n") if p.strip()]


# ---------- 加载 embedding 模型：把一段文字变成一串数字（向量） ----------
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")   # 512 维中文向量


def embed(texts):
    # normalize_embeddings=True：向量长度归一成 1，这样"点积 == 余弦相似度"
    return model.encode(texts, normalize_embeddings=True)


def cosine(a, b):
    # 两个向量都已归一化，点积就是余弦相似度（范围 -1 ~ 1，越大越相关）
    return float(a @ b)


if __name__ == "__main__":
    if DEEPSEEK_API_KEY.startswith("sk-在这里"):
        print("❌ 请先打开 config.py，把你的 API key 粘贴进去！")
        raise SystemExit(1)

    chunks = load_and_chunk("knowledge.txt")
    chunk_vecs = embed(chunks)              # (5, 512)
    print(f"已把 {len(chunks)} 个片段变成 {chunk_vecs.shape[1]} 维向量\n")

    demo = embed(["多少钱"])
    print(f"例：'多少钱' 就变成这样一个 512 维向量：{demo[0][:8]}...\n")

    while True:
        q = input("问题（q 退出）：")
        if q.strip().lower() in ("q", "quit", "exit", "退出"):
            break

        qv = embed([q])[0]                  # 问题也变成 512 维向量

        # 1. 打印余弦相似度排名（关键：看"多少钱"是不是终于排到【定价】第一）
        print("\n余弦相似度排名（越大越相关）：")
        ranked = sorted(((cosine(qv, cv), c) for cv, c in zip(chunk_vecs, chunks)),
                        reverse=True)
        for s, c in ranked:
            print(f"  {s:+.4f}  {c[:24]}...")

        # 2. 取 top-2 生成
        top = [c for _, c in ranked[:2]]
        answer = chat([
            {"role": "system",
             "content": "你是一个客服助手。只根据下面【资料】回答问题；资料里没有的，就说不知道，不要编造。\n\n【资料】\n" + "\n\n".join(top)},
            {"role": "user", "content": q},
        ])
        print("\n【带 RAG(语义)】回答：", answer)
        print("=" * 60)
