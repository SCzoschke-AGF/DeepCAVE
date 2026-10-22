FROM python:3.10-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential swig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PATH="/app/.venv/bin:${PATH}"
ENV UV_LINK_MODE=copy

# Keep dependency installation cached when only application code changes.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python /usr/local/bin/python /app/.venv \
    && uv pip install --python /app/.venv/bin/python \
        -r requirements.txt 'optuna==3.6.1'

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python \
        --no-deps --no-build-isolation -e .

EXPOSE 8050

CMD ["uv", "run", "--no-sync", "deepcave", "--config", "configs/docker.py"]
