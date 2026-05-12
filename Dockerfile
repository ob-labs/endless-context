FROM quay.io/oceanbase/seekdb:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN if command -v yum >/dev/null 2>&1; then \
      yum install -y --allowerasing curl git ca-certificates python3 python3-pip python3.12 python3.12-pip && \
      yum clean all; \
    else \
      echo "No supported package manager found." && exit 1; \
    fi

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV UV_INDEX_URL=${PIP_INDEX_URL}
ENV UV_DEFAULT_INDEX=${PIP_INDEX_URL}
ENV UV_LINK_MODE=copy
ENV UV_HTTP_TIMEOUT=300
ENV UV_HTTP_RETRIES=8
ENV UV_INDEX_STRATEGY=unsafe-first-match
ENV UV_NATIVE_TLS=true

RUN INSECURE_HOST="${PIP_INDEX_URL#http://}" && \
    if [ "${INSECURE_HOST}" != "${PIP_INDEX_URL}" ]; then \
      INSECURE_HOST="${INSECURE_HOST%%/*}"; \
      INSECURE_HOST="${INSECURE_HOST##*@}"; \
      INSECURE_HOST="${INSECURE_HOST%%:*}"; \
    else \
      INSECURE_HOST=""; \
    fi && \
    if [ -n "${INSECURE_HOST}" ]; then \
      python3.12 -m pip install --no-cache-dir -U uv -i "${PIP_INDEX_URL}" --trusted-host "${INSECURE_HOST}"; \
    else \
      python3.12 -m pip install --no-cache-dir -U uv -i "${PIP_INDEX_URL}"; \
    fi

ENV PATH="/app/.venv/bin:/root/.local/bin:${PATH}"
ENV BUB_HOME=/app/.bub
ENV BUB_WORKSPACE_PATH=/app
ENV BUB_MCP_CONFIG_PATH=/app/.bub/mcp.json

WORKDIR /app

RUN mkdir -p /app/.bub /app/.agents

COPY pyproject.toml uv.lock README.md ./
COPY app.py ./
COPY src ./src
COPY scripts ./scripts
COPY .agents/mcp.json ./.agents/mcp.json
COPY .agents/skills ./.agents/skills
COPY .env.example ./.env.example
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh
RUN uv venv --python python3.12 /app/.venv
RUN INSECURE_HOST="${PIP_INDEX_URL#http://}" && \
    if [ "${INSECURE_HOST}" != "${PIP_INDEX_URL}" ]; then \
      INSECURE_HOST="${INSECURE_HOST%%/*}"; \
      INSECURE_HOST="${INSECURE_HOST##*@}"; \
      INSECURE_HOST="${INSECURE_HOST%%:*}"; \
    else \
      INSECURE_HOST=""; \
    fi && \
    if [ -n "${INSECURE_HOST}" ]; then \
      UV_INSECURE_HOST="${INSECURE_HOST}" UV_PROJECT_ENVIRONMENT=/app/.venv uv sync --frozen --python python3.12 --no-dev --no-cache --native-tls --default-index "${PIP_INDEX_URL}"; \
    else \
      UV_PROJECT_ENVIRONMENT=/app/.venv uv sync --frozen --python python3.12 --no-dev --no-cache --native-tls --default-index "${PIP_INDEX_URL}"; \
    fi

ENV BUB_GRADIO_HOST=0.0.0.0
ENV BUB_GRADIO_PORT=7860
ENV BUB_TAPESTORE_SQLALCHEMY_URL=mysql+oceanbase://root@127.0.0.1:2881/bub

EXPOSE 7860
EXPOSE 2881

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
