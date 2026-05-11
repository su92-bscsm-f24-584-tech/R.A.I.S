
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import utility as util
label_map = {0: "negative", 1: "neutral", 2: "positive"}


# =====================================
# 1. LOAD SENTIMENT MODEL
# =====================================
def load_sentiment_model(model_path, device):


    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

    model.to(device)
    model.eval()

    return model, tokenizer

# =====================================
# 2. RUN SENTIMENT (Fetch + Predict)
# =====================================
def run_sentiment(company, model, tokenizer, device, api_key):

    news = util.fetch_news(company, api_key)

    # 🔥 SAFETY FIX
    if not news:
        return {
            "sentiment_signal": 0.0,
            "intensity": 0.0,
            "counts": {"positive": 0, "neutral": 0, "negative": 0},
            "sen_inputs": []
        }

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    articles = {}
    sen_inputs = []

    for i, n in enumerate(news):

        title = n.get("title") or ""
        desc = n.get("description") or ""

        text = f"{title} {desc}".strip()

        sen_inputs.append(text)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)

        probs = torch.softmax(out.logits, dim=1)[0]

        pred = torch.argmax(probs).item()
        conf = probs[pred].item()

        label = ["negative", "neutral", "positive"][pred]
        counts[label] += 1

        articles[str(i)] = {"sentiment": {"confidence": conf}}

    total = sum(counts.values())

    polarity = (counts["positive"] - counts["negative"]) / total if total else 0
    intensity = (counts["positive"] + counts["negative"]) / total if total else 0

    conf = sum(a["sentiment"]["confidence"] for a in articles.values()) / max(len(articles), 1)

    return {
        "sen_inputs": sen_inputs,
        "sentiment_signal": polarity * conf,
        "intensity": intensity,
        "counts": counts
    }