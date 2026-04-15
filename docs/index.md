# Endless Context Docs

## Overview

Endless Context is a Bub extension package focused on one deployment shape:

- Bub runtime and CLI stay upstream
- Gradio is exposed as a Bub channel
- tape persistence is delegated to `bub-tapestore-sqlalchemy`
- OceanBase/seekdb support is provided through a thin compatibility plugin
- the original three-pane Gradio UI/UX is preserved above the new runtime shape

This keeps the project close to the upstream Bub extension model and removes the old app-local runtime hacks.

## Architecture

- **Bub plugin**: `src/endless_context/plugin.py`
- **Gradio channel**: `src/endless_context/channel.py`
- **OceanBase compatibility**: `src/endless_context/oceanbase.py`
- **ModelScope launcher**: `app.py`

## Runtime flow

1. `app.py` creates `BubFramework`, loads installed Bub plugins, and starts `ChannelManager` with `gradio` enabled.
2. `GradioChannel` launches a Gradio `Blocks` UI on port `7860`.
3. User input is converted into `ChannelMessage` and handed to Bub through the channel message handler.
4. Bub runs the normal hook pipeline and persists tape entries through `bub-tapestore-sqlalchemy`.
5. Assistant output is routed back to `GradioChannel.send()` and rendered in the UI.

## Quick start

### Docker Compose (recommended for local)

```bash
cp .env.example .env
make compose-up
```

Starts SeekDB + app together. Stop with `make compose-down`. UI at `http://localhost:7860`.
SeekDB is kept on the Compose network by default; only the Gradio UI is published to the host.

### Single container

```bash
docker build -t endless-context:latest .
docker run --rm -p 7860:7860 -p 2881:2881 endless-context:latest
```

### ModelScope Docker Studio

- Use the provided `Dockerfile` and `docker/entrypoint.sh`.
- Exposed ports: `7860` (Gradio channel) and `2881` (SeekDB).
- Set at least `BUB_MODEL`, `BUB_API_KEY`, `BUB_API_BASE` when needed, and `BUB_TAPESTORE_SQLALCHEMY_URL`.
- Build and run the container, then open the forwarded `7860` port.

## Configuration (.env)

- `BUB_MODEL`, `BUB_API_KEY`, `BUB_API_BASE`
- `BUB_TAPESTORE_SQLALCHEMY_URL`
- `BUB_GRADIO_HOST`, `BUB_GRADIO_PORT`
- `OCEANBASE_HOST`, `OCEANBASE_PORT`, `OCEANBASE_USER`, `OCEANBASE_PASSWORD`, `OCEANBASE_DATABASE`

## Development workflow (Makefile)

- `make install` — `uv sync` + `uv run prek install`
- `make compose-up|down|logs` — Docker Compose lifecycle (local recommended)
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
