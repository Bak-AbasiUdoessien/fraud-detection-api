"""
Tests. Run from the project folder:  python -m pytest -v
A small model is trained once into a temporary folder, then the API is tested against it.
"""

import importlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient            # calls the API in-process, no server needed

from fraud import features, synthetic, train

GOOD_TX = {"step": 2, "type": "TRANSFER", "amount": 181000.0, "oldbalanceOrg": 181000.0,
           "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0}
NORMAL_TX = {"step": 14, "type": "PAYMENT", "amount": 1500.0, "oldbalanceOrg": 90000.0,
             "newbalanceOrig": 88500.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0}


@pytest.fixture(scope="module")                       # scope="module": train ONCE for all tests in this file
def model_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    train.train(synthetic.generate(rows=30_000, seed=1), out_dir=out)
    return out


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    from fraud import api
    importlib.reload(api)                             # re-read MODEL_DIR from the environment
    with TestClient(api.app) as c:                    # "with" runs the start-up code that loads the model
        yield c


# ---------- features ---------------------------------------------------------

def test_features_shape_and_order():
    f = features.build_features(pd.DataFrame([GOOD_TX]))
    assert list(f.columns) == features.FEATURES       # same columns, same order, every time
    assert f.loc[0, "orig_emptied"] == 1 and f.loc[0, "type_TRANSFER"] == 1
    assert f.loc[0, "hour"] == 2


def test_risk_signals():
    signals = features.risk_signals(GOOD_TX)
    assert "sender's account was emptied" in signals
    assert features.risk_signals({**NORMAL_TX, "newbalanceDest": 1500.0}) == []   # ** copies the dict, then overrides


def test_synthetic_has_paysim_columns():
    df = synthetic.generate(rows=2_000)
    assert list(df.columns) == features.COLUMNS
    assert 0 < df["isFraud"].mean() < 0.05            # fraud is rare


# ---------- training ---------------------------------------------------------

def test_threshold_is_a_probability(model_dir):
    meta = pd.read_json(model_dir / "metadata.json", typ="series")
    assert 0 < meta["threshold"] < 1
    assert meta["test_metrics"]["pr_auc"] > 0.3       # clearly better than random (random = the fraud rate)


# ---------- API --------------------------------------------------------------

def test_health(client):
    assert client.get("/health").json()["model_loaded"] is True


def test_predict_risky_vs_normal(client):
    risky = client.post("/predict", json=GOOD_TX).json()
    normal = client.post("/predict", json=NORMAL_TX).json()
    assert risky["fraud_probability"] > normal["fraud_probability"]
    assert risky["risk_signals"]                      # non-empty list of reasons


def test_validation_rejects_bad_input(client):
    assert client.post("/predict", json={**GOOD_TX, "amount": -5}).status_code == 422       # negative amount
    assert client.post("/predict", json={**GOOD_TX, "type": "BITCOIN"}).status_code == 422  # unknown type
    assert client.post("/predict", json={**GOOD_TX, "newbalanceOrig": 999999999}).status_code == 422


def test_batch(client):
    out = client.post("/predict/batch", json=[GOOD_TX, NORMAL_TX]).json()
    assert len(out) == 2
    assert client.post("/predict/batch", json=[]).status_code == 422


def test_no_model_gives_503(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))    # an empty folder: no model in it
    from fraud import api
    importlib.reload(api)
    with TestClient(api.app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json=GOOD_TX).status_code == 503
