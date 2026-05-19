# Endless Context Docs

Endless Context is a Bub extension package for one deployment shape: keep the Bub runtime upstream, expose Gradio as a Bub channel, and persist tape data through `bub-tapestore-sqlalchemy` with OceanBase/seekdb compatibility.

> [Bub](https://github.com/bubbuild/bub) is the upstream runtime and extension model behind Endless Context. If you are also interested in the database-native agent harness we are building, see [AgentSeek](https://github.com/ob-labs/agentseek).

## Overview

The repository is split into a few small pieces:

- `app.py` boots `BubFramework`, loads installed Bub hooks, and starts `ChannelManager` with `gradio` enabled.
- `src/endless_context/channel.py` provides the Gradio channel and UI wiring.
- `src/endless_context/plugin.py` provides the Bub plugin hooks.
- `src/endless_context/oceanbase.py` registers the OceanBase dialect expected by the tape store.

## Architecture

- **Launcher**: `app.py`
- **Plugin hooks**: `src/endless_context/plugin.py`
- **Gradio channel**: `src/endless_context/channel.py`
- **OceanBase compatibility**: `src/endless_context/oceanbase.py`

## Runtime flow

1. `app.py` creates `BubFramework`, loads installed Bub plugins, and starts `ChannelManager` with `gradio` enabled.
2. `GradioChannel` launches a Gradio `Blocks` UI on port `7860`.
3. User input is converted into `ChannelMessage` and handed to Bub through the channel message handler.
4. Bub runs the normal hook pipeline and persists tape entries through `bub-tapestore-sqlalchemy`.
5. Assistant output is routed back to `GradioChannel.send()` and rendered in the UI.

## Deployment options

### Docker Compose (recommended for local)

```bash
cp .env.example .env
make compose-up
```

Fill provider credentials in `.env` before the first start.
Starts SeekDB + app together. Stop with `make compose-down`. UI at `http://localhost:7860`.
SeekDB is kept on the Compose network by default; only the Gradio UI is published to the host.

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

Use `--env-file .env` so the container receives the same environment variables as Docker Compose.

### Bare-metal

```bash
uv sync
cp .env.example .env
make run
```

Set `BUB_TAPESTORE_SQLALCHEMY_URL` to a reachable SeekDB endpoint before starting the app.

### ModelScope Docker Studio

- Use the provided `Dockerfile` and `docker/entrypoint.sh`.
- Exposed ports: `7860` (Gradio channel) and `2881` (SeekDB).
- Set at least `BUB_MODEL`, `BUB_API_KEY`, `BUB_API_BASE` when needed, and `BUB_TAPESTORE_SQLALCHEMY_URL`.
- Build and run the container, then open the forwarded `7860` port.

## Configuration

The main runtime variables are:

- `BUB_MODEL`, `BUB_API_KEY`, `BUB_API_BASE`
- `BUB_TAPESTORE_SQLALCHEMY_URL`
- `BUB_GRADIO_HOST`, `BUB_GRADIO_PORT`
- `BUB_WORKSPACE_PATH`, `BUB_MCP_CONFIG_PATH`

`BUB_TAPESTORE_SQLALCHEMY_URL` is the single database configuration source.
The checked-in `.env.example` points at `seekdb` for Docker Compose; for bare-metal or the single-container image, change it to the actual reachable endpoint.

`BUB_WORKSPACE_PATH` defaults to `/app` inside the image.
`BUB_MCP_CONFIG_PATH` defaults to `/app/.bub/mcp.json`.
The packaged `.agents/mcp.json` is copied into that default path only after the placeholder MCP id has been replaced.
Use `--env-file .env` for single-container runs so the entrypoint receives the same environment variables as Docker Compose.
Project skills are discovered from `${BUB_WORKSPACE_PATH}/.agents/skills`.

## Development workflow (Makefile)

- `make install` — `uv sync` + `uv run prek install`
- `make compose-up` / `make compose-down` / `make compose-logs` — Docker Compose lifecycle
- `make docker-build` — build single-container image for ModelScope
- `make run` — start Bub's `ChannelManager` with the `gradio` channel via `python app.py`
- `make test` — `uv run pytest`
- `make check` — lock consistency + `prek run -a`
- `make lint` / `make fmt` — ruff check/format
- `make docs-test` / `make docs` — build/serve docs

## Testing

`pytest` now focuses on:

- Gradio channel request/response wiring and preserved UI interactions
- tape snapshot / anchor selection logic on the Bub-native tape service
- OceanBase/seekdb compatibility patch behavior

## Risks and mitigations

- External DB dependency: SeekDB must be reachable; use Docker Compose or Docker Studio when possible.
- Bub version alignment: this package targets the `0.3.x` extension model; keep the `bub` pin and entry-point contract in sync.
