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

It provides:

- a `gradio` Bub channel for browser chat
- OceanBase/seekdb dialect registration compatible with `bub-tapestore-sqlalchemy`
- a ModelScope-friendly `app.py` that boots Bub's `ChannelManager` with `gradio` enabled
- the original three-pane Gradio UI/UX for tape, conversation, and anchors

The old private `AppRuntime`, custom tape store, and monkey patches inside the app layer have been removed.

## Run on ModelScope Docker Studio

1) Keep the provided `Dockerfile` and `docker/entrypoint.sh`.
2) Exposed ports: `7860` (Gradio) and `2881` (SeekDB). Entry file is `app.py`.
3) Set environment secrets in Studio, e.g. `BUB_MODEL`, `BUB_API_KEY`, `BUB_API_BASE`, and `BUB_TAPESTORE_SQLALCHEMY_URL`.
4) Build and run. `app.py` starts Bub's channel manager with `gradio` enabled, so opening the forwarded `7860` port reaches the Bub channel directly.

## Run locally (preferred: Docker)

### Docker Compose (app + SeekDB)
```bash
cp .env.example .env   # fill in keys
make compose-up        # builds and starts everything
```
The UI is at `http://localhost:7860`. In Compose mode, SeekDB is only exposed on the internal Docker network so local port conflicts on `2881` do not block startup. Stop with `make compose-down`.
The app service uses an explicit `BUB_TAPESTORE_SQLALCHEMY_URL` that points at the `seekdb` container.

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

Use `--env-file .env` so the single-container entrypoint receives the same environment variables as Docker Compose. `BUB_MCP_CONFIG_PATH` defaults to `/app/.bub/mcp.json`; skills are discovered from `${BUB_WORKSPACE_PATH}/.agents/skills`, with `BUB_WORKSPACE_PATH` defaulting to `/app`. The image also includes `.agents/mcp.json` at `/app/.agents/mcp.json`; the entrypoint copies it to the default MCP path only after the placeholder MCP id has been replaced.

### Bare-metal (advanced, no containers)
```bash
uv sync
cp .env.example .env
make run
```
For bare-metal, set `BUB_TAPESTORE_SQLALCHEMY_URL` to the actual reachable SeekDB endpoint before starting the app.

### Bub CLI shape

`app.py` starts the same shape you would get from:

```bash
uv run bub gateway --enable-channel gradio
```

That keeps the runtime aligned with Bub's extension model instead of a project-local forked runtime.

## Docs

- `docs/index.md` contains architecture, local/Docker workflows, and configuration details.

## License

Apache License 2.0

## Related

- Bub: https://github.com/PsiACE/bub
- pyobvector: https://github.com/oceanbase/pyobvector
- SeekDB: https://www.oceanbase.ai/product/seekdb
- OceanBase: https://www.oceanbase.com/
