FROM python:3.10-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential swig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN uv venv --python /usr/local/bin/python /app/.venv \
    && uv pip install --python /app/.venv/bin/python -e '.[optuna]'

ENV PATH="/app/.venv/bin:${PATH}"
EXPOSE 8050

CMD ["uv", "run", "--no-sync", "deepcave", "--config", "configs/docker.py"]
