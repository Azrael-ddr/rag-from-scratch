# -*- coding: utf-8 -*-
r"""
phase4_eval.py —— 第四步：评估（给 RAG 出张考卷）

一套 (问题, 正确片段, 期望关键词) 的测试集，分别用
  Phase 2 字符检索  和  Phase 3 语义检索  各跑一遍，对比：
    1. 检索命中率 —— 正确片段有没有进 top-2
    2. 答案质量   —— 生成的答案里有没有出现期望关键词

运行： .\.venv\Scripts\python.exe phase4_eval.py
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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
              "temperature": 0.3, "max_tokens": 300},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def load_and_chunk(path):
    text = open(path, encoding="utf-8").read()
    return [p.strip() for p in text.split("\n\n") if p.strip()]


# —— Phase 2 的字符检索 ——
def retrieve_char(chunks, query, top_k=2):
    q_chars = set(query)
    return sorted(chunks, key=lambda c: len(q_chars & set(c)), reverse=True)[:top_k]


# —— Phase 3 的语义检索 ——
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")


def embed(texts):
    return model.encode(texts, normalize_embeddings=True)


def retrieve_sem(chunks, chunk_vecs, query, top_k=2):
    qv = embed([query])[0]
    order = sorted(range(len(chunks)), key=lambda i: float(qv @ chunk_vecs[i]), reverse=True)
    return [chunks[i] for i in order[:top_k]]


# ---------- 测试集：(问题, 正确片段标题, 期望关键词) ----------
# gold_title=None 表示"文档里没有，应回答不知道"
EVAL = [
    ("NovaChat 企业版多少钱？", "【定价】", "299"),
    ("多少钱",                 "【定价】", "299"),
    ("价格是多少",             "【定价】", "299"),
    ("星辉科技总部在哪？",     "【公司简介】", "杭州"),
    ("怎么退款？",             "【售后服务】", "7 天"),
    ("怎么联系商务合作？",     "【联系方式】", "partner@stellartech.com"),
    ("NovaChat 能接入微信吗？", "【产品介绍】", "微信"),
    ("NovaChat 支持哪些语言？", None, "不知道"),
]


def hit(top_chunks, gold_title):
    # 有明确正确答案的片段：检查它有没有进 top-2
    return gold_title is None or any(gold_title in c for c in top_chunks)


if __name__ == "__main__":
    chunks = load_and_chunk("knowledge.txt")
    chunk_vecs = embed(chunks)

    char_hit = sem_hit = 0
    retr_n = 0
    ans_ok = 0

    for i, (q, gold_title, gold_kw) in enumerate(EVAL, 1):
        c_top = retrieve_char(chunks, q)
        s_top = retrieve_sem(chunks, chunk_vecs, q)

        c_ok = hit(c_top, gold_title)
        s_ok = hit(s_top, gold_title)
        if gold_title is not None:      # 有正确答案的才计入检索命中率
            retr_n += 1
            char_hit += c_ok
            sem_hit += s_ok

        # 生成（用语义检索的 top-2）
        answer = chat([
            {"role": "system",
             "content": "你是一个客服助手。只根据下面【资料】回答问题；资料里没有的，就说不知道，不要编造。\n\n【资料】\n" + "\n\n".join(s_top)},
            {"role": "user", "content": q},
        ])
        kw_ok = gold_kw in answer
        ans_ok += kw_ok

        print(f"Q{i}: {q}")
        print(f"    字符检索:{'✓' if c_ok else '✗'}  语义检索:{'✓' if s_ok else '✗'}  答案含'{gold_kw}':{'✓' if kw_ok else '✗'}")

    print("\n检索命中率   字符 {}/{}  |  语义 {}/{}".format(char_hit, retr_n, sem_hit, retr_n))
    print("答案质量(含期望关键词) {}/{}".format(ans_ok, len(EVAL)))
