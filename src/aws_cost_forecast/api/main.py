"""FastAPI inference API: serves the trained TimeSeriesTransformer.

The checkpoint is loaded once, at application startup (`lifespan`), and
injected into the endpoints via `Depends` — this allows testing the
endpoints by swapping the model for a test double, without needing a
real trained checkpoint on disk.
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
            "Window of historical daily costs, in chronological order. "
            "Must have the same size as the window used to train the model."
        ),
    )
    steps: int = Field(
        default=7, ge=1, le=90, description="Number of days ahead to forecast."
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
            detail="Model not loaded yet. Train the model and generate a checkpoint first.",
        )

    if len(request.historical_costs) != bundle.input_window:
        raise HTTPException(
            status_code=422,
            detail=(
                f"historical_costs must contain exactly {bundle.input_window} "
                "values (the model's window size)."
            ),
        )

    forecast_values = forecast_from_history(
        bundle.model, bundle.scaler, request.historical_costs, request.steps, bundle.device
    )
    return ForecastResponse(forecast=forecast_values, steps=request.steps)
