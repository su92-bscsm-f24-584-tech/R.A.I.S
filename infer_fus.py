def fuse(lstm_out, sent, alpha=0.3):

    # 1. Normalize LSTM direction to probability space
    lstm_signal = lstm_out["direction"] * lstm_out["confidence"]  # [-1, 1]

    # 2. Normalize sentiment impact safely
    sentiment_impact = sent["sentiment_signal"] * sent["intensity"]  # [-1, 1]

    # 3. Combine in weighted form (NOT multiplicative chaos)
    raw_score = (1 - alpha) * lstm_signal + alpha * sentiment_impact

    # 4. Convert to probability (0–1 confidence)
    confidence = (raw_score + 1) / 2

    # 5. Clamp for safety
    confidence = max(0, min(1, confidence))

    return {
        "lstm_score": lstm_signal,
        "sentiment_score": sentiment_impact,
        "final_score": confidence
    }