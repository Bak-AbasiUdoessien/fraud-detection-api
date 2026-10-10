"""
api.py  -  A REST API that scores transactions for fraud.

    uvicorn fraud.api:app --reload          # then open http://127.0.0.1:8000/docs

FastAPI builds interactive documentation automatically at /docs from the code below.
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator   # data validation library used by FastAPI

from .features import build_features, risk_signals

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
state = {}                                                 # holds the loaded model while the server runs


@asynccontextmanager
async def lifespan(app):
    """Runs once at start-up: load the model into memory (not on every request - that would be slow)."""
    try:
        state["model"] = joblib.load(MODEL_DIR / "model.joblib")
        state["meta"] = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        state.clear()                                      # start anyway; /health will say the model is missing
    yield                                                  # the server handles requests while paused here
    state.clear()                                          # runs at shutdown


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan,
              description="Scores mobile-money transactions for fraud risk. Learning project - not for real decisions.")


class Transaction(BaseModel):
    """One transaction. Field rules are checked automatically; bad input gets a clear 422 error."""
    step: int = Field(..., ge=0, description="hours since the start of the period", examples=[3])
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] = Field(..., examples=["TRANSFER"])
    amount: float = Field(..., gt=0, examples=[181000.0])           # gt = greater than
    oldbalanceOrg: float = Field(..., ge=0, examples=[181000.0])    # ge = greater than or equal
    newbalanceOrig: float = Field(..., ge=0, examples=[0.0])
    oldbalanceDest: float = Field(..., ge=0, examples=[0.0])
    newbalanceDest: float = Field(..., ge=0, examples=[0.0])

    @model_validator(mode="after")                          # an extra check across several fields
    def sender_cannot_gain(self):
        if self.newbalanceOrig > self.oldbalanceOrg + self.amount:
            raise ValueError("newbalanceOrig is larger than the old balance plus the amount")
        return self


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    risk_signals: list[str]


def require_model():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: python -m fraud.train")


def score(transactions):
    """Score a list of Transaction objects in one go (faster than one by one)."""
    rows = pd.DataFrame([t.model_dump() for t in transactions])    # model_dump() -> plain dict
    proba = state["model"].predict_proba(build_features(rows))[:, 1]
    threshold = state["meta"]["threshold"]
    return [Prediction(fraud_probability=round(float(p), 4), is_fraud=bool(p >= threshold),
                       threshold=threshold, risk_signals=risk_signals(t.model_dump()))
            for p, t in zip(proba, transactions)]


@app.get("/health")
def health():
    """Is the service up, and is a model loaded? (Used by Docker and cloud hosts to check the app.)"""
    return {"status": "ok", "model_loaded": "model" in state,
            "model": state.get("meta", {}).get("model")}


@app.get("/model")
def model_info():
    """Which model is running, what it was trained on, and how well it scored on held-out test data."""
    require_model()
    return state["meta"]


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction):
    """Score one transaction."""
    require_model()
    return score([tx])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(txs: list[Transaction]):
    """Score up to 1,000 transactions in one request."""
    require_model()
    if not 1 <= len(txs) <= 1000:
        raise HTTPException(status_code=422, detail="send between 1 and 1000 transactions")
    return score(txs)
