# syntax=docker/dockerfile:1

# Build stage: installs dependencies via uv (using the lockfile, without
# the dev group) and assembles the virtualenv with the package installed.
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev

# Final stage: only the ready virtualenv + source code, no build
# toolchain, smaller image.
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    CHECKPOINT_PATH="/app/checkpoints/model.pt"

EXPOSE 8000

CMD ["uvicorn", "aws_cost_forecast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
