import yfinance as yf
import torch
import torch.nn as nn


# =====================================
# 1. LOAD MODEL
# =====================================
import torch
import torch.nn as nn

def load_lstm_model(model_path, device, input_size=5):

    class StockLSTM(nn.Module):
        def __init__(self):
            super().__init__()

            self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
            self.lstm2 = nn.LSTM(64, 32, batch_first=True)

            self.shared_fc = nn.Linear(32, 16)
            self.relu = nn.ReLU()

            self.regression_head = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )

            self.direction_head = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            x, _ = self.lstm1(x)
            x, _ = self.lstm2(x)

            x = x[:, -1, :]
            x = self.relu(self.shared_fc(x))

            reg = self.regression_head(x)
            price, ret = reg[:, 0:1], reg[:, 1:2]

            d = self.direction_head(x)
            conf = torch.sigmoid(torch.abs(d))

            return price, ret, d, conf


    model = StockLSTM()

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model

# =====================================
# 2. RUN LSTM (Fetch + Predict)
# =====================================
def run_lstm(company, model, device, window=30, period="6mo"):

    import yfinance as yf
    import numpy as np
    import torch

    df = yf.download(company, period=period)

    # ✔ USE ALL 5 FEATURES (VERY IMPORTANT)
    data = df[["Open", "High", "Low", "Close", "Volume"]].dropna().values

    # normalize
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

    # last window
    seq = data[-window:]

    # ✔ CORRECT SHAPE: (1, window, 5)
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

    # sanity check (optional debug)
    print("INPUT SHAPE:", seq.shape)

    with torch.no_grad():
        price, ret, direction, confidence = model(seq)

    return {
        "price": price.item(),
        "return": ret.item(),
        "direction": torch.sigmoid(direction).item(),
        "confidence": confidence.item()
    }