# -*- coding: utf-8 -*-
r"""
phase2_rag.py —— 第二步：RAG 最小骨架（检索 + 生成）

模型本身不知道 knowledge.txt 里写了什么。
RAG 的思路：先把和问题相关的文档片段「检索」出来，塞进 prompt，再让模型「生成」答案。

流程：切块(chunk) → 关键词检索(retrieve) → 组装 prompt → 生成(generate)

运行： .\.venv\Scripts\python.exe phase2_rag.py
"""
import requests
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


# ---------- 1. 读文档 + 切块 ----------
def load_and_chunk(path):
    text = open(path, encoding="utf-8").read()
    # 空行分段，每段就是一个"块"
    return [p.strip() for p in text.split("\n\n") if p.strip()]


# ---------- 2. 关键词检索（Phase 2 用最简单的字符重叠，Phase 3 换成 embedding） ----------
def retrieve(chunks, query, top_k=2):
    q_chars = set(query)

    def score(chunk):
        # 问题里的字，有多少个出现在这个块里
        return len(q_chars & set(chunk))

    ranked = sorted(chunks, key=score, reverse=True)
    return ranked[:top_k]


# ---------- 3. 组装 prompt + 生成 ----------
def rag_answer(chunks, query):
    hits = retrieve(chunks, query)
    context = "\n\n".join(hits)
    messages = [
        {"role": "system",
         "content": "你是一个客服助手。只根据下面【资料】回答问题；资料里没有的，就说不知道，不要编造。\n\n【资料】\n" + context},
        {"role": "user", "content": query},
    ]
    return chat(messages), hits


if __name__ == "__main__":
    if DEEPSEEK_API_KEY.startswith("sk-在这里"):
        print("❌ 请先打开 config.py，把你的 API key 粘贴进去！")
        raise SystemExit(1)

    chunks = load_and_chunk("knowledge.txt")
    print(f"已把 knowledge.txt 切成 {len(chunks)} 个片段\n")

    while True:
        q = input("问题（q 退出）：")
        if q.strip().lower() in ("q", "quit", "exit", "退出"):
            break

        # —— 对比 1：不带 RAG，直接问模型 ——
        print("\n【不带 RAG】", end="", flush=True)
        print(chat([{"role": "system",
                     "content": "你是一个乐于助人的助手，不确定就说不确定，不要编造。"},
                    {"role": "user", "content": q}]))

        # —— 对比 2：带 RAG ——
        answer, hits = rag_answer(chunks, q)
        print("\n【带 RAG】检索到的片段：")
        for i, h in enumerate(hits, 1):
            print(f"  片段{i}：{h[:50]}...")
        print("【带 RAG】回答：", end="", flush=True)
        print(answer)
        print("=" * 60)
