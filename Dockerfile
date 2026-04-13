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
ENV UV_HTTP_RETRIES=3
ENV UV_INDEX_STRATEGY=unsafe-first-match

RUN python3.12 -m pip install --no-cache-dir -U uv -i "${PIP_INDEX_URL}"

ENV PATH="/app/.venv/bin:/root/.local/bin:${PATH}"
ENV BUB_HOME=/app/.bub
ENV BUB_WORKSPACE_PATH=/app

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY app.py ./
COPY src ./src
COPY scripts ./scripts
COPY .env.example ./.env.example
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/setup-bub-workspace.py /usr/local/bin/setup-bub-workspace.py

RUN chmod +x /usr/local/bin/entrypoint.sh
RUN uv venv --python python3.12 /app/.venv
RUN UV_PROJECT_ENVIRONMENT=/app/.venv uv sync --frozen --python python3.12 --no-dev --default-index "${PIP_INDEX_URL}"

ENV BUB_GRADIO_HOST=0.0.0.0
ENV BUB_GRADIO_PORT=7860
ENV OCEANBASE_HOST=127.0.0.1
ENV OCEANBASE_PORT=2881

EXPOSE 7860
EXPOSE 2881

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
