"""
=====================================================================
 LSTM Price Prediction Model — Multi-Asset Portfolio
=====================================================================
Assets   : BTC, ETH (CoinGecko) | RIOT, Crude Oil (Yahoo Finance)
Model    : Long Short-Term Memory (LSTM) — Deep Learning
Horizon  : 30-day price forecast per asset
Author   : <O teu nome>
Date     : 2026
=====================================================================
Dependências:
    pip install requests pandas numpy matplotlib seaborn scikit-learn tensorflow yfinance
=====================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import time
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ──────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────
LOOKBACK     = 60     # dias de histórico para prever 1 dia
FORECAST     = 30     # dias a prever no futuro
EPOCHS       = 100    # máximo de epochs (early stopping activo)
BATCH_SIZE   = 32
TRAIN_SPLIT  = 0.80   # 80% treino / 20% teste
OUTPUT_DIR   = "."

PALETTE = {
    "BTC":       "#F7931A",
    "ETH":       "#627EEA",
    "RIOT":      "#E63946",
    "Crude Oil": "#2D6A4F",
}

# ──────────────────────────────────────────────
# 1. RECOLHA DE DADOS
# ──────────────────────────────────────────────
def fetch_crypto(coin_id: str, ticker: str, days: int = 365) -> pd.Series:
    """Busca histórico diário de cripto via CoinGecko."""
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
           f"?vs_currency=usd&days={days}&interval=daily")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["prices"], columns=["ts", "price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    df = df.drop_duplicates("date").set_index("date")["price"]
    df.name = ticker
    time.sleep(1.5)
    return df

def fetch_yahoo(ticker: str, label: str, period: str = "2y") -> pd.Series:
    """Busca histórico diário de acções/commodities via Yahoo Finance."""
    data = yf.download(ticker, period=period, interval="1d", progress=False)
    s = data["Close"].squeeze()
    s.index = pd.to_datetime(s.index).normalize()
    s.name = label
    return s

def load_all_data() -> dict:
    print("📡 A recolher dados...\n")
    assets = {}

    # Cripto
    for coin_id, ticker in [("bitcoin", "BTC"), ("ethereum", "ETH")]:
        try:
            s = fetch_crypto(coin_id, ticker)
            assets[ticker] = s
            print(f"  ✅ {ticker} — {len(s)} dias (CoinGecko)")
        except Exception as e:
            print(f"  ❌ {ticker} — {e}")

    # Acções / Commodities
    for yahoo_ticker, label in [("RIOT", "RIOT"), ("CL=F", "Crude Oil")]:
        try:
            s = fetch_yahoo(yahoo_ticker, label)
            assets[label] = s
            print(f"  ✅ {label} — {len(s)} dias (Yahoo Finance)")
        except Exception as e:
            print(f"  ❌ {label} — {e}")

    return assets

# ──────────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO
# ──────────────────────────────────────────────
def prepare_sequences(series: pd.Series, lookback: int, train_split: float):
    """
    Normaliza a série, cria sequências X/y para o LSTM
    e divide em treino/teste.
    """
    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)   # [samples, timesteps, features]

    split = int(len(X) * train_split)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler, scaled

# ──────────────────────────────────────────────
# 3. MODELO LSTM
# ──────────────────────────────────────────────
def build_lstm(lookback: int) -> Sequential:
    """Arquitectura LSTM com duas camadas e Dropout para regularização."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_model(model, X_train, y_train):
    es = EarlyStopping(monitor="val_loss", patience=10,
                       restore_best_weights=True, verbose=0)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.10,
        callbacks=[es],
        verbose=0
    )
    return history

# ──────────────────────────────────────────────
# 4. PREVISÃO
# ──────────────────────────────────────────────
def forecast_future(model, scaled, scaler, lookback: int, days: int) -> np.ndarray:
    """Previsão recursiva para os próximos N dias."""
    last_seq = scaled[-lookback:].reshape(1, lookback, 1)
    preds = []
    for _ in range(days):
        pred = model.predict(last_seq, verbose=0)[0, 0]
        preds.append(pred)
        last_seq = np.roll(last_seq, -1, axis=1)
        last_seq[0, -1, 0] = pred
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

def evaluate(model, X_test, y_test, scaler):
    preds_scaled = model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    mae  = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mape = np.mean(np.abs((actual - preds) / actual)) * 100
    return preds, actual, mae, rmse, mape

# ──────────────────────────────────────────────
# 5. VISUALIZAÇÕES
# ──────────────────────────────────────────────
def plot_prediction(asset_name: str, series: pd.Series,
                    actual_test: np.ndarray, preds_test: np.ndarray,
                    future_dates: pd.DatetimeIndex, future_preds: np.ndarray,
                    mae: float, rmse: float, mape: float):

    color = PALETTE.get(asset_name, "#555")
    n_test = len(actual_test)
    test_dates = series.index[-n_test:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{asset_name} — LSTM Price Prediction",
                 fontsize=16, fontweight="bold", y=0.98)

    # ── Top: full chart ─────────────────────────────────────────────
    ax = axes[0]
    split_idx = len(series) - n_test

    ax.plot(series.index[:split_idx], series.values[:split_idx],
            color="#94A3B8", linewidth=1.2, label="Training Data", alpha=0.7)
    ax.plot(test_dates, actual_test,
            color=color, linewidth=2, label="Actual Price (Test)")
    ax.plot(test_dates, preds_test,
            color="black", linewidth=1.5, linestyle="--", label="LSTM Prediction (Test)")

    # Forecast zone
    ax.fill_betweenx([min(future_preds)*0.92, max(future_preds)*1.08],
                     future_dates[0], future_dates[-1],
                     color=color, alpha=0.07)
    ax.plot(future_dates, future_preds,
            color=color, linewidth=2.5, linestyle="-.",
            marker="o", markersize=3, label=f"30-Day Forecast")

    ax.axvline(series.index[-1], color="grey", linestyle=":", linewidth=1, alpha=0.8)
    ax.annotate("Forecast →", xy=(series.index[-1], series.values[-1]),
                xytext=(10, 0), textcoords="offset points",
                fontsize=9, color="grey")

    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)

    # Metrics box
    metrics_text = f"MAE: ${mae:,.2f}   RMSE: ${rmse:,.2f}   MAPE: {mape:.2f}%"
    ax.text(0.99, 0.04, metrics_text, transform=ax.transAxes,
            fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CBD5E1", alpha=0.9))

    # ── Bottom: residuals ────────────────────────────────────────────
    ax2 = axes[1]
    residuals = actual_test - preds_test
    ax2.bar(test_dates, residuals,
            color=[color if r >= 0 else "#EF4444" for r in residuals],
            alpha=0.7, width=1)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Residuals (Actual − Predicted)", fontsize=10)
    ax2.set_ylabel("USD", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)
    ax2.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax2)

    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{asset_name.replace(' ', '_')}_lstm_forecast.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 Gráfico guardado: {fname}")


def plot_summary_dashboard(results: dict):
    """Dashboard comparativo com as previsões de todos os activos."""
    n = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle("30-Day Price Forecast — Multi-Asset LSTM Dashboard",
                 fontsize=15, fontweight="bold")

    for i, (asset, res) in enumerate(results.items()):
        ax = axes[i]
        color = PALETTE.get(asset, "#555")

        # Last 90 days of actual
        series = res["series"]
        ax.plot(series.index[-90:], series.values[-90:],
                color=color, linewidth=2, label="Recent Price")
        ax.plot(res["future_dates"], res["future_preds"],
                color="black", linewidth=2, linestyle="--",
                marker="o", markersize=3, label="30-Day Forecast")

        ax.fill_between(res["future_dates"],
                        res["future_preds"] * 0.95,
                        res["future_preds"] * 1.05,
                        color=color, alpha=0.15, label="±5% Confidence Band")

        last_price  = series.values[-1]
        final_price = res["future_preds"][-1]
        change_pct  = ((final_price - last_price) / last_price) * 100
        direction   = "▲" if change_pct >= 0 else "▼"
        dir_color   = "#16A34A" if change_pct >= 0 else "#DC2626"

        ax.set_title(f"{asset}", fontsize=12, fontweight="bold", color=color)
        ax.text(0.98, 0.95,
                f"{direction} {abs(change_pct):.1f}% in 30d",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                color=dir_color, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=dir_color, alpha=0.9))
        ax.text(0.98, 0.05,
                f"MAPE: {res['mape']:.1f}%",
                transform=ax.transAxes, fontsize=8,
                color="#6B7280", ha="right", va="bottom")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
        ax.set_ylabel("Price (USD)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/0_dashboard_forecast.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  💾 Dashboard guardado: 0_dashboard_forecast.png")


# ──────────────────────────────────────────────
# 6. RELATÓRIO FINAL
# ──────────────────────────────────────────────
def print_final_report(results: dict):
    print("\n" + "="*65)
    print("  LSTM FORECAST REPORT — 30-DAY OUTLOOK")
    print("="*65)
    print(f"{'Asset':<12} {'Current':>12} {'30d Forecast':>14} {'Change':>10} {'MAPE':>8}")
    print("-"*65)
    for asset, res in results.items():
        last   = res["series"].values[-1]
        future = res["future_preds"][-1]
        chg    = ((future - last) / last) * 100
        arrow  = "▲" if chg >= 0 else "▼"
        print(f"{asset:<12} ${last:>11,.2f}  ${future:>12,.2f}  "
              f"{arrow}{abs(chg):>7.1f}%  {res['mape']:>6.1f}%")
    print("="*65)
    print("\n⚠️  DISCLAIMER: These forecasts are for educational purposes only.")
    print("    LSTM models do not account for news, regulations or black swan events.")
    print("    Do NOT use this as financial advice.\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("="*65)
    print("  🤖 LSTM Multi-Asset Price Prediction")
    print("="*65)

    # 1. Dados
    raw_data = load_all_data()

    results = {}

    for asset_name, series in raw_data.items():
        print(f"\n{'─'*50}")
        print(f"  🔧 Training LSTM for {asset_name}...")
        print(f"{'─'*50}")

        series = series.dropna().sort_index()

        if len(series) < LOOKBACK * 2:
            print(f"  ⚠️  Not enough data for {asset_name}, skipping.")
            continue

        # 2. Sequências
        X_train, X_test, y_train, y_test, scaler, scaled = prepare_sequences(
            series, LOOKBACK, TRAIN_SPLIT)

        # 3. Modelo
        model = build_lstm(LOOKBACK)
        print(f"  📐 Architecture: 2× LSTM (128→64) + Dropout + Dense")
        print(f"  📊 Training samples: {len(X_train)} | Test samples: {len(X_test)}")

        history = train_model(model, X_train, y_train)
        epochs_run = len(history.history["loss"])
        print(f"  ✅ Training complete ({epochs_run} epochs, early stopping)")

        # 4. Avaliação
        preds_test, actual_test, mae, rmse, mape = evaluate(
            model, X_test, y_test, scaler)
        print(f"  📈 MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f} | MAPE: {mape:.2f}%")

        # 5. Forecast
        future_preds = forecast_future(model, scaled, scaler, LOOKBACK, FORECAST)
        last_date    = series.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1), periods=FORECAST, freq="D")

        # 6. Gráfico individual
        plot_prediction(asset_name, series, actual_test, preds_test,
                        future_dates, future_preds, mae, rmse, mape)

        results[asset_name] = {
            "series":       series,
            "future_dates": future_dates,
            "future_preds": future_preds,
            "mape":         mape,
        }

    # 7. Dashboard comparativo
    if results:
        print(f"\n{'─'*50}")
        print("  📊 Generating summary dashboard...")
        plot_summary_dashboard(results)
        print_final_report(results)

    print("\n✅ All done! Check the generated charts in the current folder.\n")
