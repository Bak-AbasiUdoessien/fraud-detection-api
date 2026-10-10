"""
synthetic.py  -  MADE-UP mobile-money transactions in the PaySim column format, so the
project runs without downloading anything. The patterns are invented (loosely inspired by
how PaySim's fraud behaves: fraud happens in TRANSFER and CASH_OUT and often empties the
sender's account). Swap in the real PaySim CSV for real results - see README.

    python -m fraud.synthetic --rows 200000
"""

import argparse

import numpy as np                                   # fast random numbers and arrays
import pandas as pd

from .features import COLUMNS                        # the PaySim column names we must produce

TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
TYPE_SHARE = [0.34, 0.08, 0.35, 0.22, 0.01]          # how common each type is (sums to 1)


def generate(rows=200_000, fraud_rate=0.006, seed=42):
    """Return a DataFrame of `rows` transactions, about `fraud_rate` of them fraudulent."""
    rng = np.random.default_rng(seed)                # a seeded generator -> same data every run
    n_fraud = int(rows * fraud_rate)
    n_legit = rows - n_fraud

    # ---- legitimate transactions --------------------------------------------------------
    legit = pd.DataFrame({
        "step": rng.integers(0, 744, n_legit),                       # hour of a 31-day month (PaySim's unit)
        "type": rng.choice(TYPES, n_legit, p=TYPE_SHARE),
        "amount": np.round(rng.lognormal(mean=9.5, sigma=1.3, size=n_legit), 2),   # many small, a few large
        "oldbalanceOrg": np.round(rng.lognormal(10.5, 1.6, n_legit), 2),
        "oldbalanceDest": np.round(rng.lognormal(11, 2.0, n_legit) * rng.binomial(1, 0.7, n_legit), 2),  # 30% start at 0
    })
    legit["amount"] = np.minimum(legit["amount"], legit["oldbalanceOrg"] * rng.uniform(0.05, 1.0, n_legit))
    legit["amount"] = legit["amount"].round(2).clip(lower=1)        # at least 1 unit
    # Honest behaviour that LOOKS like fraud, so the model can't rely on one simple rule:
    drains = rng.random(n_legit) < 0.04                                 # 4% of honest people send everything they have
    legit.loc[drains, "amount"] = legit.loc[drains, "oldbalanceOrg"]
    legit["newbalanceOrig"] = (legit["oldbalanceOrg"] - legit["amount"]).clip(lower=0).round(2)
    lag = rng.random(n_legit) < 0.03                                    # 3%: receiver's balance not updated yet
    legit["newbalanceDest"] = np.where(lag, legit["oldbalanceDest"],
                                       legit["oldbalanceDest"] + legit["amount"]).round(2)
    legit["isFraud"] = 0

    # ---- fraudulent transactions --------------------------------------------------------
    old = np.round(rng.lognormal(11.5, 1.4, n_fraud), 2)
    empties = rng.random(n_fraud) < 0.70                               # 70% drain the whole account
    amount = np.where(empties, old, np.round(old * rng.uniform(0.3, 0.95, n_fraud), 2))
    night = rng.random(n_fraud) < 0.5                                  # half happen between 00:00 and 05:00
    hour = np.where(night, rng.integers(0, 6, n_fraud), rng.integers(0, 24, n_fraud))
    fraud = pd.DataFrame({
        "step": rng.integers(0, 31, n_fraud) * 24 + hour,
        "type": rng.choice(["TRANSFER", "CASH_OUT"], n_fraud),
        "amount": amount.clip(min=1),
        "oldbalanceOrg": old,
        "oldbalanceDest": np.where(rng.random(n_fraud) < 0.6, 0.0, np.round(rng.lognormal(11, 2, n_fraud), 2)),
    })
    fraud["newbalanceOrig"] = (fraud["oldbalanceOrg"] - fraud["amount"]).clip(lower=0).round(2)
    mule_hides = rng.random(n_fraud) < 0.5                             # mule accounts often don't show the money arriving
    fraud["newbalanceDest"] = np.where(mule_hides, fraud["oldbalanceDest"],
                                       fraud["oldbalanceDest"] + fraud["amount"]).round(2)
    fraud["isFraud"] = 1

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=seed)   # shuffle rows
    df["nameOrig"] = "C" + pd.Series(rng.integers(1e8, 1e9, len(df))).astype(str).values  # fake account ids
    df["nameDest"] = "M" + pd.Series(rng.integers(1e8, 1e9, len(df))).astype(str).values
    df["isFlaggedFraud"] = 0
    return df.sort_values("step").reset_index(drop=True)[COLUMNS]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=200_000)
    p.add_argument("--out", default="data/synthetic_transactions.csv")
    a = p.parse_args()
    df = generate(a.rows)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} MADE-UP transactions ({df['isFraud'].mean():.2%} fraud) -> {a.out}")


if __name__ == "__main__":
    main()
