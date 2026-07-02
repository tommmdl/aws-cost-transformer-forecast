# AWS Cost Transformer Forecast

> Quando FinOps encontra Deep Learning: prevendo custo de nuvem com Transformers.

## Pitch

Como Senior Cloud Operations Analyst com foco em FinOps, lido diariamente com previsão e otimização de
custo AWS para 13+ contas enterprise. Este projeto aplica uma arquitetura Transformer — implementada do
zero (Positional Encoding, attention, treinamento) durante minha pós-graduação em Matemática e
Estatística Aplicada para Data Science, ML e IA — para prever séries temporais de custo cloud, unindo
dois mundos: expertise de domínio em FinOps e Deep Learning aplicado.

## Status

🚧 Em desenvolvimento — ver [Definition of Done](#definition-of-done).

## Stack

PyTorch · FastAPI · Docker · uv · pytest

## Estrutura

```
aws-cost-transformer-forecast/
├── src/aws_cost_forecast/
│   ├── data/synthetic_aws_cost.py   # gerador da série sintética de custo AWS
│   ├── model/                       # PositionalEncoding + TimeSeriesTransformer
│   ├── training/train.py            # script de treino via CLI
│   └── api/main.py                  # FastAPI app (/forecast, /health)
├── notebooks/                       # treino + avaliação + gráficos
├── tests/                           # pytest
└── docs/arquitetura.md              # a matemática por trás (PE, attention)
```

## Como rodar

```bash
uv sync
uv run pytest
uv run uvicorn aws_cost_forecast.api.main:app --reload
```

## Definition of Done

- [ ] `uv run pytest` passa 100%
- [ ] `docker compose up` sobe a API e `/forecast` responde com uma previsão coerente
- [ ] Notebook roda do início ao fim sem erro e gera os gráficos de previsão
- [ ] README explica o pitch, a arquitetura e tem pelo menos 1 imagem/gráfico
- [ ] Nenhum dado real de cliente ou da empresa em lugar nenhum do repo
