"""
=====================================================================
 Ensemble Price Prediction — LSTM + XGBoost + Random Forest
=====================================================================
Assets   : BTC, ETH, AAPL, EUR/USD
Features : Close + RSI + MACD + Bollinger Bands + Volume + Momentum
Target   : Next day close price
Author   : Rodrigo Ferreira Alves
Date     : 2026
=====================================================================
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow yfinance
=====================================================================
"""

import warnings
warnings.filterwarnings("ignore")
import os

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ──────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────
OUTPUT_DIR  = "."
LOOKBACK    = 20
TEST_DAYS   = 60
EPOCHS      = 50
BATCH_SIZE  = 16

ASSETS = {
    "BTC":    {"ticker": "BTC-USD",  "color": "#F7931A"},
    "ETH":    {"ticker": "ETH-USD",  "color": "#627EEA"},
    "AAPL":   {"ticker": "AAPL",     "color": "#A8B5C1"},
    "EURUSD": {"ticker": "EURUSD=X", "color": "#2D6A4F"},
}

# ──────────────────────────────────────────────
# 1. RECOLHA DE DADOS
# ──────────────────────────────────────────────
def load_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                  for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["open","high","low","close","volume"]].dropna()
    return df

# ──────────────────────────────────────────────
# 2. FEATURES TÉCNICAS
# ──────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c  = df["close"]

    df["sma_10"] = c.rolling(10).mean()
    df["sma_20"] = c.rolling(20).mean()
    df["ema_12"] = c.ewm(span=12, adjust=False).mean()
    df["ema_26"] = c.ewm(span=26, adjust=False).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    bb_mid         = c.rolling(20).mean()
    bb_std         = c.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-10)

    df["ret_1"]  = c.pct_change(1)
    df["ret_5"]  = c.pct_change(5)
    df["ret_10"] = c.pct_change(10)
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(10).mean() + 1e-10)
    df["hl_pct"]    = (df["high"] - df["low"]) / (c + 1e-10)
    df["co_pct"]    = (c - df["open"]) / (df["open"] + 1e-10)

    df["target"] = c.shift(-1)

    return df.dropna()

FEATURES = [
    "close","sma_10","sma_20","ema_12","ema_26",
    "rsi","macd","macd_signal","macd_hist",
    "bb_upper","bb_lower","bb_pct","bb_width",
    "ret_1","ret_5","ret_10","vol_10",
    "vol_ratio","hl_pct","co_pct"
]

# ──────────────────────────────────────────────
# 3. PREPARAR DADOS
# ──────────────────────────────────────────────
def prepare(df: pd.DataFrame):
    df    = add_features(df)
    n     = len(df)
    print(f"    Linhas após features: {n}")

    split = n - TEST_DAYS
    split = max(split, LOOKBACK + 30)
    split = min(split, n - LOOKBACK - 10)

    X = df[FEATURES].values
    y = df["target"].values
    dates = df.index

    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    d_te       = dates[split:]

    print(f"    Train: {len(X_tr)} | Test: {len(X_te)}")

    sx = MinMaxScaler()
    sy = MinMaxScaler()
    X_tr_s = sx.fit_transform(X_tr)
    X_te_s = sx.transform(X_te)
    y_tr_s = sy.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_te_s = sy.transform(y_te.reshape(-1,1)).ravel()

    return X_tr_s, X_te_s, y_tr_s, y_te_s, y_tr, y_te, d_te, sx, sy

def make_seq(X, y, lb=LOOKBACK):
    Xs, ys = [], []
    for i in range(lb, len(X)):
        Xs.append(X[i-lb:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# ──────────────────────────────────────────────
# 4. MODELOS
# ──────────────────────────────────────────────
def train_lstm(X_tr, y_tr, X_te, y_te):
    Xtr_s, ytr_s = make_seq(X_tr, y_tr)
    Xte_s, yte_s = make_seq(X_te, y_te)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, X_tr.shape[1])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xtr_s, ytr_s,
              validation_data=(Xte_s, yte_s),
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=[EarlyStopping(patience=10,
                         restore_best_weights=True, verbose=0)],
              verbose=0)
    return model.predict(Xte_s, verbose=0).ravel(), yte_s

def train_xgb(X_tr, y_tr, X_te, y_te):
    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0, early_stopping_rounds=20, eval_metric="rmse")
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return model.predict(X_te), model

def train_rf(X_tr, y_tr, X_te):
    model = RandomForestRegressor(
        n_estimators=200, max_depth=8,
        min_samples_leaf=3, n_jobs=-1, random_state=42)
    model.fit(X_tr, y_tr)
    return model.predict(X_te), model

# ──────────────────────────────────────────────
# 5. ENSEMBLE
# ──────────────────────────────────────────────
def ensemble(lstm_p, xgb_p, rf_p, y_te_s, sy):
    off    = len(xgb_p) - len(lstm_p)
    xgb_al = xgb_p[off:]
    rf_al  = rf_p[off:]
    y_al   = y_te_s[off:]

    def dn(p): return sy.inverse_transform(p.reshape(-1,1)).ravel()

    lr, xr, rr, yr = dn(lstm_p), dn(xgb_al), dn(rf_al), dn(y_al)

    m   = [mean_absolute_error(yr, p) for p in [lr, xr, rr]]
    inv = [1/x for x in m]
    tot = sum(inv)
    w   = [x/tot for x in inv]

    print(f"    Pesos — LSTM:{w[0]*100:.1f}%  XGB:{w[1]*100:.1f}%  RF:{w[2]*100:.1f}%")

    ens = w[0]*lr + w[1]*xr + w[2]*rr
    return ens, yr, lr, xr, rr, w

# ──────────────────────────────────────────────
# 6. MÉTRICAS
# ──────────────────────────────────────────────
def calc_metrics(y, p, name=""):
    mae  = mean_absolute_error(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    mape = np.mean(np.abs((y - p) / (y + 1e-10))) * 100
    da   = np.mean(np.sign(np.diff(y)) == np.sign(np.diff(p))) * 100
    print(f"    {name:<18} MAE={mae:>10,.1f}  RMSE={rmse:>10,.1f}  "
          f"MAPE={mape:>5.1f}%  DirAcc={da:.1f}%")
    return dict(mae=mae, rmse=rmse, mape=mape, da=da)

# ──────────────────────────────────────────────
# 7. GRÁFICOS
# ──────────────────────────────────────────────
def plot_pred(dates, y, ens, lstm, xgb_p, rf_p, asset, color):
    n = len(ens)
    d = dates[-n:]
    fig, axes = plt.subplots(2, 1, figsize=(14,9),
                             gridspec_kw={"height_ratios":[3,1]})
    fig.suptitle(f"{asset} — Ensemble Prediction (LSTM + XGBoost + RF)",
                 fontsize=13, fontweight="bold")
    ax = axes[0]
    ax.plot(d, y[-n:],  color="white",   lw=2,   label="Actual")
    ax.plot(d, ens,     color=color,     lw=2.5, label="Ensemble")
    ax.plot(d, lstm,    color="#94A3B8", lw=1,   ls="--", label="LSTM",    alpha=.7)
    ax.plot(d, xgb_p,  color="#F59E0B", lw=1,   ls="--", label="XGBoost", alpha=.7)
    ax.plot(d, rf_p,   color="#10B981", lw=1,   ls="--", label="RF",      alpha=.7)
    ax.fill_between(d, ens*.98, ens*1.02, alpha=.1, color=color)
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    sns.despine(ax=ax)
    ax2 = axes[1]
    err  = ens - y[-n:]
    cols = ["#22C55E" if e >= 0 else "#EF4444" for e in err]
    ax2.bar(d, err, color=cols, alpha=.8, width=1.5)
    ax2.axhline(0, color="white", lw=.8)
    ax2.set_ylabel("Error")
    ax2.grid(axis="y", alpha=.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)
    sns.despine(ax=ax2)
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{asset}_ensemble.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 {fname}")

def plot_importance(rf_model, asset):
    imp = pd.Series(rf_model.feature_importances_,
                    index=FEATURES).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(9,6))
    imp.plot(kind="barh", ax=ax, color="#7C3AED", alpha=.85)
    ax.set_title(f"{asset} — Feature Importance (Random Forest)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{asset}_importance.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 {fname}")

def plot_summary(all_m):
    assets = list(all_m.keys())
    models = ["LSTM","XGBoost","RF","Ensemble"]
    colors = ["#94A3B8","#F59E0B","#10B981","#F7931A"]
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    fig.suptitle("Model Comparison — All Assets", fontsize=13, fontweight="bold")
    x = np.arange(len(assets))
    w = 0.2
    for i, (metric, label, ax) in enumerate(zip(
            ["mape","da"], ["MAPE (%)","Directional Accuracy (%)"], axes)):
        for j, (model, col) in enumerate(zip(models, colors)):
            vals = [all_m[a].get(model,{}).get(metric,0) for a in assets]
            ax.bar(x + j*w, vals, w, label=model, color=col, alpha=.85)
        ax.set_xticks(x + w*1.5)
        ax.set_xticklabels(assets)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=.3)
        sns.despine(ax=ax)
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/0_summary.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 {fname}")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  Ensemble Model — LSTM + XGBoost + Random Forest")
    print("="*60)

    all_metrics = {}

    for asset, cfg in ASSETS.items():
        print(f"\n{'─'*50}")
        print(f"  {asset}")
        print(f"{'─'*50}")

        try:
            df = load_data(cfg["ticker"])
            print(f"  ✅ {len(df)} dias")
        except Exception as e:
            print(f"  ❌ {e}"); continue

        (X_tr, X_te, y_tr_s, y_te_s,
         y_tr, y_te, d_te, sx, sy) = prepare(df)

        print("  🔧 LSTM...")
        lstm_p, lstm_true = train_lstm(X_tr, y_tr_s, X_te, y_te_s)

        print("  🔧 XGBoost...")
        xgb_p, xgb_m = train_xgb(X_tr, y_tr_s, X_te, y_te_s)

        print("  🔧 Random Forest...")
        rf_p, rf_m = train_rf(X_tr, y_tr_s, X_te)

        ens, y_r, lstm_r, xgb_r, rf_r, w = ensemble(
            lstm_p, xgb_p, rf_p, y_te_s, sy)

        print(f"\n  Métricas:")
        all_metrics[asset] = {
            "LSTM":    calc_metrics(y_r, lstm_r, "LSTM"),
            "XGBoost": calc_metrics(y_r, xgb_r,  "XGBoost"),
            "RF":      calc_metrics(y_r, rf_r,    "Random Forest"),
            "Ensemble":calc_metrics(y_r, ens,     "Ensemble ⭐"),
        }

        plot_pred(d_te, y_r, ens, lstm_r, xgb_r, rf_r, asset, cfg["color"])
        plot_importance(rf_m, asset)

    plot_summary(all_metrics)

    print("\n" + "="*60)
    print("  RELATÓRIO FINAL")
    print("="*60)
    for asset, ms in all_metrics.items():
        print(f"\n  {asset}")
        for model, m in ms.items():
            star = " ⭐" if model == "Ensemble" else ""
            print(f"    {model:<12} MAPE={m['mape']:.1f}%  DirAcc={m['da']:.1f}%{star}")
    print("\n✅ Concluído!\n")
