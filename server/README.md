# Mem0 REST API Server

Mem0 provides a REST API server (written using FastAPI). Users can perform all operations through REST endpoints. The API also includes OpenAPI documentation, accessible at `/docs` when the server is running.

## Features

- **Create memories:** Create memories based on messages for a user, agent, or run.
- **Retrieve memories:** Get all memories for a given user, agent, or run.
- **Search memories:** Search stored memories based on a query.
- **Update memories:** Update an existing memory.
- **Delete memories:** Delete a specific memory or all memories for a user, agent, or run.
- **Reset memories:** Reset all memories for a user, agent, or run.
- **OpenAPI Documentation:** Accessible via `/docs` endpoint.

## Running the server

Follow the instructions in the [docs](https://docs.mem0.ai/open-source/features/rest-api) to run the server.

## 自定义 LLM / Embedder（DashScope 通义千问示例）

通过环境变量可切换为阿里云 DashScope（OpenAI 兼容模式）等第三方模型，用于摘要生成（LLM）和向量化（Embedder）。

1. 复制 `.env.example` 为 `.env`，按需填写。
2. 使用 DashScope 时，在 `.env` 中配置例如：

```env
OPENAI_API_KEY=sk-your-dashscope-api-key

# 摘要生成模型（LLM）
MEM0_LLM_MODEL=qwen-plus
MEM0_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MEM0_LLM_TEMPERATURE=0.2

# 向量模型（Embedder）
MEM0_EMBEDDER_MODEL=text-embedding-v4
MEM0_EMBEDDER_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MEM0_EMBEDDER_DIMENSION=1024
```

不设置 `MEM0_*` 时，将使用默认 OpenAI 配置（`gpt-4.1-nano-2025-04-14`、`text-embedding-3-small`）。底层逻辑见 `mem0/memory/main.py` 中 `Memory.from_config` 及 `search` / `add` 使用的 LLM 与 Embedder。
