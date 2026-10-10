"""
features.py  -  Turn raw transaction fields into model features, and into human-readable
risk signals. Used by BOTH training and the API, so the model always sees features
built exactly the same way (avoids "training/serving skew", a classic ML production bug).
"""

import numpy as np
import pandas as pd

# The PaySim dataset's columns (also what synthetic.py produces).
COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
           "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

FEATURES = ["amount", "log_amount", "hour", "amount_to_balance",
            "orig_balance_error", "dest_balance_error", "orig_emptied", "dest_was_empty",
            *[f"type_{t}" for t in TYPES]]                     # * unpacks the list into this one


def build_features(df):
    """Return a DataFrame with exactly the FEATURES columns, in order."""
    f = pd.DataFrame(index=df.index)                          # same row labels as the input
    f["amount"] = df["amount"]
    f["log_amount"] = np.log1p(df["amount"])                  # log1p(x) = log(1 + x): tames huge amounts
    f["hour"] = df["step"] % 24                               # PaySim's step = hours since the start
    f["amount_to_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1)        # +1 avoids dividing by zero
    # In an honest transaction the books balance, so these two "errors" are near 0.
    f["orig_balance_error"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    f["dest_balance_error"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    f["orig_emptied"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)   # True/False -> 1/0
    f["dest_was_empty"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)).astype(int)
    for t in TYPES:
        f[f"type_{t}"] = (df["type"] == t).astype(int)        # one-hot encoding: one 0/1 column per type
    return f[FEATURES]


def risk_signals(tx):
    """
    Plain-English reasons a single transaction (a dict) looks risky. These are simple
    rules for the analyst reading the result - they explain the INPUT, not the model's maths.
    """
    signals = []
    if tx["oldbalanceOrg"] > 0 and tx["newbalanceOrig"] == 0:
        signals.append("sender's account was emptied")
    if tx["oldbalanceOrg"] > 0 and tx["amount"] >= 0.9 * tx["oldbalanceOrg"]:
        signals.append("amount is at least 90% of the sender's balance")
    if abs(tx["oldbalanceDest"] + tx["amount"] - tx["newbalanceDest"]) > 0.01 * max(tx["amount"], 1):
        signals.append("receiver's balance doesn't add up")
    if tx["oldbalanceDest"] == 0 and tx["newbalanceDest"] == 0:
        signals.append("receiving account shows no balance before or after")
    if tx["step"] % 24 < 6:
        signals.append("made between midnight and 6am")
    if tx["type"] in ("TRANSFER", "CASH_OUT"):
        signals.append(f"{tx['type']} - the type most fraud uses")
    return signals
