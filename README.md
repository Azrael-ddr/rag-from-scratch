# rag-from-scratch

从零手写 **RAG（检索增强生成）**，不调用任何现成 RAG 框架。

一个"私有知识库问答"系统的最小实现：把文档切成块 → 检索出和问题最相关的片段 → 塞进大模型 prompt 生成答案。用 4 个阶段从最简对话逐步构建到完整 RAG，并用测试集量化对比了两种检索方式。

## 核心结论

**RAG 的上限 = 检索质量。** 生成模型再强，检索喂错片段，答案就错（garbage in, garbage out）。

字符级关键词检索在"换了个说法"的问题上会失效（如问"多少钱"却匹配不到"月费 299 元"），语义检索靠 embedding + 余弦相似度解决这个问题。

## 4 阶段递进

| 阶段 | 文件 | 内容 |
|---|---|---|
| Phase 1 | `phase1_chat.py` | 最简对话：HTTP 直接调 DeepSeek API |
| Phase 2 | `phase2_rag.py` | RAG 最小骨架：切块 + 字符检索 + 生成 |
| Phase 3 | `phase3_embedding.py` | 语义检索：embedding + 余弦相似度 |
| Phase 4 | `phase4_eval.py` | 评估：测试集量化对比两种检索 |

## 评估结果（8 题测试集）

| 指标 | 结果 |
|---|---|
| 字符检索命中率 | 5 / 7 |
| **语义检索命中率** | **7 / 7** |
| **答案质量（含期望信息）** | **8 / 8** |

## 快速运行

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

在项目根目录新建 `config.py`（已被 .gitignore 排除，不会上传）：

```python
DEEPSEEK_API_KEY = "sk-你的key"
```

然后任选一个阶段运行：

```bash
.\.venv\Scripts\python.exe phase1_chat.py     # 最简对话
.\.venv\Scripts\python.exe phase2_rag.py      # 字符检索 RAG
.\.venv\Scripts\python.exe phase3_embedding.py # 语义检索 RAG（首次会下载 ~95MB 模型）
.\.venv\Scripts\python.exe phase4_eval.py     # 评估
```

> 语义检索首次运行会自动从 `hf-mirror.com` 下载 `bge-small-zh-v1.5` 模型。

## 技术栈

Python · requests · numpy · torch · sentence-transformers(bge-small-zh-v1.5) · DeepSeek API
