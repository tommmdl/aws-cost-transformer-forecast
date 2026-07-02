"""API FastAPI de inferência: serve o TimeSeriesTransformer treinado.

O checkpoint é carregado uma vez, na subida da aplicação (`lifespan`), e
injetado nos endpoints via `Depends` — isso permite testar os endpoints
substituindo o modelo por um dublê nos testes, sem precisar de um
checkpoint real treinado em disco.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from aws_cost_forecast.api.inference import ModelBundle, forecast_from_history, load_model_bundle
from aws_cost_forecast.training.train import DEFAULT_CHECKPOINT_PATH

_model_bundle: ModelBundle | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _model_bundle
    checkpoint_path = Path(os.environ.get("CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH))
    _model_bundle = load_model_bundle(checkpoint_path)
    yield


app = FastAPI(title="AWS Cost Forecast API", lifespan=lifespan)


def get_model_bundle() -> ModelBundle | None:
    return _model_bundle


class ForecastRequest(BaseModel):
    historical_costs: list[float] = Field(
        ...,
        description=(
            "Janela de custos diários históricos, em ordem cronológica. "
            "Deve ter o mesmo tamanho da janela usada no treino do modelo."
        ),
    )
    steps: int = Field(
        default=7, ge=1, le=90, description="Quantidade de dias à frente para prever."
    )


class ForecastResponse(BaseModel):
    forecast: list[float]
    steps: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health(bundle: ModelBundle | None = Depends(get_model_bundle)) -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=bundle is not None)


@app.post("/forecast", response_model=ForecastResponse)
def forecast(
    request: ForecastRequest, bundle: ModelBundle | None = Depends(get_model_bundle)
) -> ForecastResponse:
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda não carregado. Treine e gere um checkpoint primeiro.",
        )

    if len(request.historical_costs) != bundle.input_window:
        raise HTTPException(
            status_code=422,
            detail=(
                f"historical_costs deve conter exatamente {bundle.input_window} "
                "valores (tamanho da janela do modelo)."
            ),
        )

    forecast_values = forecast_from_history(
        bundle.model, bundle.scaler, request.historical_costs, request.steps, bundle.device
    )
    return ForecastResponse(forecast=forecast_values, steps=request.steps)
