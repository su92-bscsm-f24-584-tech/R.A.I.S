import numpy as np
import joblib
import torch

import lstm as ls
import emotion as sen
import lr as lr_model   # prophet brain
import utility as util

from sklearn.ensemble import RandomForestRegressor

device = torch.device("cpu")

# =========================
# LOAD BASE MODELS
# =========================
sent_model, tokenizer = sen.load_sentiment_model(util.sen_path, device)

lstm_model = ls.load_lstm_model(util.lstm_path, device)


# =========================
# BUILD DATASET
# =========================
def build_dataset(companies):

    dataset = []

    for c in companies:

        ticker = util.resolve(c)

        # ---- LSTM BRAIN ----
        lstm_out = ls.run_lstm(ticker, lstm_model, device)

        # ---- SENTIMENT BRAIN ----
        sent_out = sen.run_sentiment(
            ticker,
            sent_model,
            tokenizer,
            device,
            api_key="YOUR_API_KEY"
        )

        # ---- PROPHET / LR BRAIN ----
        lr_out = lr_model.lr_brain(ticker)

        # =========================
        # FINAL FEATURE VECTOR
        # =========================
        features = [
            lstm_out["price"],
            lstm_out["return"],
            lstm_out["direction"],
            lstm_out["confidence"],

            sent_out["sentiment_signal"],
            sent_out["intensity"],

            lr_out["lr_direction"],
            lr_out["lr_strength"]
        ]

        # target = future return (YOU ALREADY HAVE FUNCTION)
        target = util.get_future_return(ticker, days=5)

        dataset.append((features, target))

    return dataset


# =========================
# TRAIN META MODEL
# =========================
def train_meta(dataset):

    X = np.array([d[0] for d in dataset])
    y = np.array([d[1] for d in dataset])

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)

    joblib.dump(model, util.meta_path)

    print("✅ META MODEL TRAINED & SAVED")

    return model


# =========================
# RUN
# =========================
if __name__ == "__main__":

    companies = [
        "AAPL", "MSFT", "GOOGL", "TSLA",
        "AMZN", "NVDA", "META"
    ]

    dataset = build_dataset(companies)

    meta_model = train_meta(dataset)

    print("DONE")