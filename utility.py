sen_path = "C:/Users/imato/Downloads/Compressed/PYTHON PROJECTS/AI LAB PROJECT/sen_model"
lstm_path = r"C:\Users\imato\Downloads\Compressed\PYTHON PROJECTS\AI LAB PROJECT\lstm_model\lstm_model.pth"
meta_path=r"C:\Users\imato\Downloads\Compressed\PYTHON PROJECTS\AI LAB PROJECT\meta_brain\meta_model.pkl"
api="pub_5c901480c8264d9e9da1ad631362d736"
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from lstm import run_lstm
import requests
COMPANY_MAP = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "META": "Facebook / Meta",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "NFLX": "Netflix",
    "ORCL": "Oracle",
    "ADBE": "Adobe",
    "PYPL": "PayPal",
    "ABNB": "Airbnb",
    "UBER": "Uber",
    "SQ": "Square",
    "JPM": "JP Morgan / Chase",
    "BAC": "Bank of America / BofA",
    "V": "Visa",
    "MA": "Mastercard",
    "BRK-B": "Berkshire Hathaway",
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "VOO": "Vanguard S&P 500",
    "TSM": "TSMC",
    "BABA": "Alibaba",
    "TM": "Toyota",
    "SONY": "Sony",
    "ASML": "ASML",
    "SAP": "SAP",
    "DIS": "Disney",
    "MCD": "McDonald's",
    "PG": "Procter & Gamble",
    "MMM": "3M",
    "BA": "Boeing",
    "XOM": "Exxon Mobil",
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "BTC-USD": "Bitcoin"
}
def resolve(user_input: str):
    user_input = user_input.lower().strip()

    # exact ticker
    for ticker in COMPANY_MAP:
        if user_input == ticker.lower():
            return ticker

    # partial name match
    for ticker, name in COMPANY_MAP.items():
        if user_input in name.lower():
            return ticker

    # reverse search (important)
    for ticker, name in COMPANY_MAP.items():
        if name.lower() in user_input:
            return ticker
    
    return None
import yfinance as yf
import matplotlib.pyplot as plt

def stock_history(company, years=2):

    period = f"{years}y"

    df = yf.download(company, period=period)

    plt.figure(figsize=(10,5))
    plt.plot(df.index, df["Close"], label="Close Price")

    plt.title(f"{company} Stock Price - Last {years} Years")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()

    plt.show()

    return df
import numpy as np
import torch

def recursive_lstm_predict(model, initial_seq, steps, device):

    model.eval()

    seq = initial_seq.copy()
    predictions = []

    for _ in range(steps):

        x = torch.tensor(seq, dtype=torch.float32)\
                .unsqueeze(0).unsqueeze(-1).to(device)

        with torch.no_grad():
            out = model(x)

        price = out[0][0].item()  # predicted price

        predictions.append(price)

        # 🔁 shift window (recursive input)
        seq = np.append(seq[1:], price)

    return predictions
import matplotlib.pyplot as plt

def plot_recursive(predictions):

    plt.figure(figsize=(10,5))
    plt.plot(predictions, marker="o", label="Recursive Forecast")

    plt.title("LSTM Recursive Stock Prediction")
    plt.xlabel("Step")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.grid()

    plt.show()

def long_run_fetch(ticker, years=5, future_years=2):

    # download long history
    df = yf.download(ticker, period=f"{years}y")

    close = df["Close"].dropna().values

    # time index
    X = np.arange(len(close)).reshape(-1,1)
    y = close

    # trend model
    model = LinearRegression()
    model.fit(X, y)

    # future
    future_steps = int(252 * future_years)   # trading days
    future_X = np.arange(len(close), len(close)+future_steps).reshape(-1,1)

    future_pred = model.predict(future_X)

    # trend direction
    slope = model.coef_[0]

    direction = "bullish" if slope > 0 else "bearish"

    return {
        "direction": direction,
        "slope": float(slope),
        "current_price": float(close[-1]),
        "future_price": float(future_pred[-1]),
        "forecast": future_pred.tolist()
    }

def mtlstm(current_input, limit,company,lstm_model,device):
    # Base case: stop when limit hits 0
    if limit <= 0:
        return current_input
    
    # Run the model using the previous output as the new input
    # Note: adjust 'current_input' usage based on your specific run_lstm requirements
    new_output = ls.run_lstm(company, lstm_model, device, data=current_input)
    
    # Recursive call: pass the new output and decrement the limit
    return mtlstm(new_output, limit - 1)


def get_future_return(ticker, days=5):
    """
    Calculates the percentage return over the next 'days' for a given ticker.
    """
    try:
        # Download recent data (get a bit more than 'days' to be safe)
        data = yf.download(ticker, period="1mo", interval="1d", progress=False)
        
        if data.empty or len(data) < days + 1:
            return 0.0

        # Current price (most recent close)
        current_price = data['Close'].iloc[-days-1]
        
        # Future price (the latest price available)
        future_price = data['Close'].iloc[-1]
        
        # Calculate percentage return
        future_return = (future_price - current_price) / current_price
        
        # Return as a float value
        return float(future_return.squeeze())
        
    except Exception as e:
        print(f"Error calculating future return for {ticker}: {e}")
        return 0.0

def fetch_news(company, api_key, limit=10):

    url = "https://newsdata.io/api/1/news"

    params = {
        "apikey": api_key,
        "q": company,
        "language": "en",
        "category": "business"
    }

    res = requests.get(url, params=params).json()

    raw_news = res.get("results", [])

    # =========================
    # SAFE TYPE CHECK
    # =========================
    if not isinstance(raw_news, list):
        return []   # fail-safe

    return raw_news[:limit]