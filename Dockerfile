# syntax=docker/dockerfile:1

# Estágio de build: instala dependências via uv (usando o lockfile, sem
# grupo dev) e monta o virtualenv com o pacote instalado.
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev

# Estágio final: só o virtualenv pronto + código-fonte, sem toolchain de
# build, imagem menor.
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    CHECKPOINT_PATH="/app/checkpoints/model.pt"

EXPOSE 8000

CMD ["uvicorn", "aws_cost_forecast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
