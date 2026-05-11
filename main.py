import torch
import lstm as ls
import emotion as sen
import lr as lr_model
import utility as util
import infer_fus as fus
import joblib
import numpy as np

device = torch.device("cpu")

# =========================
# LOAD MODELS ONCE
# =========================
sent_model, tokenizer = sen.load_sentiment_model(util.sen_path, device)

lstm_model = ls.load_lstm_model(util.lstm_path, device)

meta_model = joblib.load(util.meta_path)


# =========================
# PIPELINE FUNCTION
# =========================
def run_pipeline(user_input):

    # 1. resolve company
    ticker = util.resolve(user_input)
    if ticker is None:
        return {"error": "Company not found"}

    # 2. LSTM brain
    lstm_out = ls.run_lstm(ticker, lstm_model, device)

    # 3. Sentiment brain
    sent_out = sen.run_sentiment(
        ticker,
        sent_model,
        tokenizer,
        device,
        api_key=util.api
    )

    # 4. Prophet / LR brain
    lr_out = lr_model.lr_brain(ticker)

    # 5. Fusion brain (optional explainability layer)
    fusion = fus.fuse(lstm_out, sent_out)

    # 6. META brain input vector
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

    features = np.array(features).reshape(1, -1)

    # 7. META prediction
    meta_prediction = meta_model.predict(features)[0]

    # 8. final response
    return {
        "ticker": ticker,
        "lstm": lstm_out,
        "sentiment": sent_out,
        "prophet": lr_out,
        "fusion": fusion,
        "meta_score": float(meta_prediction)
    }


# =========================
# CLI TEST
# =========================
if __name__ == "__main__":

    user = input("Enter company: ")

    result = run_pipeline(user)

    print("\n===== FINAL RESULT =====")
    print(result)