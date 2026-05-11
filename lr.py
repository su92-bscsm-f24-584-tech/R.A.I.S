# lr.py (Long Range Brain - Prophet)
from prophet import Prophet
import yfinance as yf
import utility as util
from prophet import Prophet
import yfinance as yf

def lr_brain(ticker, years=5, future_days=365):

    # 1. get historical data
    df = yf.download(ticker, period=f"{years}y")[["Close"]].reset_index()

    # 2. format for Prophet
    df.columns = ["ds", "y"]

    # 3. train model (statistical fitting)
    model = Prophet()
    model.fit(df)

    # 4. predict future
    future = model.make_future_dataframe(periods=future_days)
    forecast = model.predict(future)

    # 5. extract useful signals

    current = df["y"].iloc[-1]
    future_value = forecast["yhat"].iloc[-1]

    # direction signal (IMPORTANT for meta model)
    direction = 1 if future_value > current else 0

    strength = abs(future_value - current) / current

    return {
        "current_price": float(current),
        "future_price": float(future_value),
        "lr_direction": direction,
        "lr_strength": float(strength),
        "forecast_series": forecast[["ds", "yhat"]].tail(future_days)
    }
