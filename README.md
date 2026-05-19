---
# Detailed docs: https://modelscope.cn/docs/studios/create
domain:
# domain: cv/nlp/audio/multi-modal/AutoML
tags:
  - chatbot
  - memory
  - gradio
  - oceanbase
datasets:
  evaluation:
  test:
  train:
models:
# - organization/model
## Entry file for Gradio/Streamlit is app.py by default
# deployspec:
#   entry_file: app.py
license: Apache License 2.0
---

# Endless Context

A lightweight Gradio chat agent with tape-first context management powered by Bub, SeekDB, and OceanBase. Built to be ModelScope-friendly while staying easy to run locally.

> [Bub](https://github.com/bubbuild/bub) is the upstream runtime and extension model behind Endless Context. If you are also interested in the database-native agent harness we are building, see [AgentSeek](https://github.com/ob-labs/agentseek).

This repository provides:

- a `gradio` Bub channel for browser chat
- OceanBase/seekdb dialect registration compatible with `bub-tapestore-sqlalchemy`
- a ModelScope-friendly `app.py` that boots Bub's `ChannelManager` with `gradio` enabled
- the original three-pane Gradio UI/UX for tape, conversation, and anchors

The old private `AppRuntime`, custom tape store, and monkey patches inside the app layer have been removed.

## Quick start

For local work, Docker Compose is the shortest path:

```bash
cp .env.example .env
make compose-up
```

Fill provider credentials in `.env` before the first start.
Open `http://localhost:7860` after the app is ready. Stop everything with `make compose-down`.

## Run locally

### Docker Compose (recommended)

Runs the app and SeekDB together. In this mode, SeekDB stays on the internal Docker network, so host port conflicts on `2881` do not block startup.
The checked-in `.env.example` already points `BUB_TAPESTORE_SQLALCHEMY_URL` at the `seekdb` service.

### Single container

```bash
docker build -t endless-context:latest .
docker run --rm \
  --env-file .env \
  -p 7860:7860 \
  -p 2881:2881 \
  -v "$PWD/.agents/mcp.json:/app/.bub/mcp.json:ro" \
  -v "$PWD/.agents/skills:/app/.agents/skills" \
  endless-context:latest
```

Use `--env-file .env` so the container gets the same environment variables as Docker Compose.
Inside the image, `BUB_WORKSPACE_PATH` defaults to `/app` and `BUB_MCP_CONFIG_PATH` defaults to `/app/.bub/mcp.json`.
The image also includes `.agents/mcp.json` at `/app/.agents/mcp.json`; the entrypoint copies it into the default MCP path only after the placeholder MCP id has been replaced.

### Bare-metal (advanced, no containers)
```bash
uv sync
cp .env.example .env
make run
```
For bare-metal, set `BUB_TAPESTORE_SQLALCHEMY_URL` to the actual reachable SeekDB endpoint before starting the app.

## Run on ModelScope Docker Studio

1) Keep the provided `Dockerfile` and `docker/entrypoint.sh`.
2) Exposed ports: `7860` for Gradio and `2881` for SeekDB.
3) Set environment secrets in Studio, including `BUB_MODEL`, `BUB_API_KEY`, `BUB_API_BASE`, and `BUB_TAPESTORE_SQLALCHEMY_URL`.
4) Build and run, then open the forwarded `7860` port.

## Runtime shape

`app.py` starts the same shape you would get from:

```bash
uv run bub gateway --enable-channel gradio
```

That keeps the runtime aligned with Bub's extension model instead of a project-local forked runtime.

## More docs

- `docs/index.md` contains architecture, configuration, and development workflow details.

## License

Apache License 2.0

## Related

- Bub: https://github.com/bubbuild/bub
- AgentSeek: https://github.com/ob-labs/agentseek
- pyobvector: https://github.com/oceanbase/pyobvector
- SeekDB: https://www.oceanbase.ai/product/seekdb
- OceanBase: https://www.oceanbase.com/
