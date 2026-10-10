# Fraud Detection API

[![tests](https://github.com/Bak-AbasiUdoessien/fraud-detection-api/actions/workflows/tests.yml/badge.svg)](https://github.com/Bak-AbasiUdoessien/fraud-detection-api/actions/workflows/tests.yml)

A machine-learning service that scores **mobile-money transactions** for fraud risk, the kind of system that fintechs such as Paystack, Flutterwave, Moniepoint and OPay run behind every transfer. The project trains and compares models with **scikit-learn**, serves the best one through a **FastAPI** REST API with input validation, explains each result in plain English, and ships as a **Docker** image that CI builds and calls on every push.

```
POST /predict  {"type": "TRANSFER", "amount": 181000, "oldbalanceOrg": 181000, "newbalanceOrig": 0, ...}
→ {"fraud_probability": 0.99, "is_fraud": true, "threshold": 0.98,
   "risk_signals": ["sender's account was emptied", "receiver's balance doesn't add up", ...]}
```

## Run it
```bash
python -m venv .venv && .venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
python -m fraud.train                                   # trains on made-up data (~5 seconds)
uvicorn fraud.api:app --reload                          # open http://127.0.0.1:8000/docs and try it
python -m pytest -v                                     # 9 tests
```
**Docker:** `docker build -t fraud-api .` then `docker run -p 8000:8000 fraud-api`

## Endpoints
| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Is the service up, and is a model loaded? |
| GET | `/model` | Which model is running, what data it was trained on, and its test scores |
| POST | `/predict` | Score one transaction |
| POST | `/predict/batch` | Score up to 1,000 transactions |
| GET | `/docs` | Interactive documentation that FastAPI generates automatically |

## The machine-learning decisions
- **Accuracy is the wrong metric.** Fraud is under 1% of transactions, so "never fraud" is 99% accurate and useless. Models are compared on **PR-AUC**, and the report also shows **precision**, **recall**, **% of fraud money caught** and **alerts per 1,000 transactions** (the workload for the fraud team).
- **Three-way split:** train (60%), validation (20%) for choosing the model and the alert threshold, and test (20%) used **once** for the final scores. All three are stratified so each has the same fraud rate.
- **Class weighting** (`class_weight="balanced"`) so the rare fraud cases aren't ignored.
- **Model comparison:** a scaled logistic regression against histogram gradient boosting, with the better one on validation PR-AUC kept.
- **Threshold tuning:** the alert cut-off maximises F1 on validation data instead of defaulting to 0.5.
- **One feature module** (`features.py`) is used by both training and the API, so the model sees identical features in production. This avoids a classic bug called *training/serving skew*.
- **Engineered features:** balance errors (do the books add up?), whether the account was emptied, amount relative to balance, hour of day, and one-hot transaction type.

## Results (made-up data, so read the note below)
| Metric (held-out test set) | Value |
|---|---|
| PR-AUC | 0.65 |
| Precision at chosen threshold | 0.74 |
| Recall at chosen threshold | 0.53 |
| Share of fraud **money** caught | 75% |
| Alerts per 1,000 transactions | 4.3 |

These numbers come from **synthetic data whose patterns I invented** (`fraud/synthetic.py`), so they show the method working, not real-world performance. Train on the real dataset to get real numbers.

## Train on the real PaySim dataset
1. Download **"Synthetic Financial Datasets For Fraud Detection" (PaySim)** from Kaggle. It has 6.3 million mobile-money transactions modelled on real logs from an African mobile-money service.
2. Put the CSV in `data/` (git ignores it) and run:
   `python -m fraud.train --data data/PS_20174392719_1491204439457_log.csv --sample 1000000`
3. Restart the API. `/model` now reports the PaySim scores, which you can put in this README.

## Deploy (free tier)
On **Render.com**: New → Web Service → connect this repo → Runtime: **Docker** → Deploy. Render builds the Dockerfile, which trains the model during the build, and gives you a public URL ending in `/docs`.

## Limitations
- The risk signals are simple rules that explain the **input**, not the model's internal reasoning. SHAP values would be the next step.
- There's no authentication or rate limiting yet, which a real deployment would need before going public.
- The split is random rather than by time. Real fraud systems should train on the past and test on the future.

**Learning project.** Don't use it to block real payments.
