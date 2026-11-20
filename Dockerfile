FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

# dependencies first so a code change does not reinstall torch
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY models ./models
RUN uv sync --frozen --no-dev

# the trained checkpoint and the gallery live outside the image:
# docker run --gpus all -v $PWD/runs:/app/runs -v $PWD/data/gallery:/app/data/gallery lsj-biometrics identificar --foto ...
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["lsj"]
CMD ["--help"]
