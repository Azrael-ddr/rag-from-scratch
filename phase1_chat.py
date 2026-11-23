# -*- coding: utf-8 -*-
r"""
phase1_chat.py —— 第一步：看懂「AI 软件是怎么回答问题的」

这里不用任何框架，直接发一个 HTTP 请求给 DeepSeek 的大模型 API。
整个 ChatGPT 类应用的底层，就是这一件事：
    把对话发过去 → 模型返回一段文本。

运行：  .\.venv\Scripts\python.exe phase1_chat.py
前提：  先在 config.py 里填好 DEEPSEEK_API_KEY
"""
import requests
from config import DEEPSEEK_API_KEY

# DeepSeek 的接口地址（OpenAI 兼容格式）
URL = "https://api.deepseek.com/chat/completions"


def chat(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    # 注意 messages：一个列表，每项有 role（system=给模型定人设，user=你说的话）
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个乐于助人的助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,   # 采样温度：越大越随机，越小越确定
        "max_tokens": 500,    # 最多回复多少个 token
    }
    r = requests.post(URL, headers=headers, json=body, timeout=60)
    r.raise_for_status()      # 出错就抛异常，方便看错误信息
    data = r.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    if DEEPSEEK_API_KEY.startswith("sk-在这里"):
        print("❌ 请先打开 config.py，把你的 API key 粘贴进去！")
        raise SystemExit(1)

    print("输入问题回车，输入 exit / q 退出。\n")
    while True:
        q = input("你：")
        if q.strip().lower() in ("exit", "quit", "q", "退出"):
            break
        print("AI：", end="", flush=True)
        print(chat(q))
