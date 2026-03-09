"""
=====================================================================
 Order Flow + Volume Profile Analysis — NQ & ES Futures
=====================================================================
Assets     : NQ (Nasdaq 100 Futures) + ES (S&P 500 Futures)
Components : VWAP | Volume Profile (POC, VAH, VAL) | Cumulative Delta
Timeframe  : 1h intraday (2 years) + daily (5 years)
Author     : Rodrigo Ferreira Alves
Date       : 2026
=====================================================================
pip install pandas numpy matplotlib seaborn yfinance
=====================================================================

DEFINIÇÕES:
  VWAP  = Volume Weighted Average Price
          Benchmark institucional — preço justo ponderado pelo volume.
          Preço acima VWAP = bullish bias | abaixo = bearish bias.

  Volume Profile = distribuição de volume por nível de preço.
          POC (Point of Control) = nível com mais volume transaccionado.
          VAH (Value Area High)  = limite superior da Value Area (70% do volume).
          VAL (Value Area Low)   = limite inferior da Value Area (70% do volume).

  Cumulative Delta = diferença acumulada entre volume de compra e venda.
          Estimativa: candle bullish (close>open) = buy volume
                      candle bearish (close<open) = sell volume
          Delta crescente com preço = confirmação de tendência.
          Divergência delta/preço   = possível reversão.
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta

OUTPUT_DIR = "."

# ──────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────
ASSETS = {
    "NQ":  {"ticker": "NQ=F",    "name": "Nasdaq 100 Futures", "color": "#627EEA"},
    "ES":  {"ticker": "ES=F",    "name": "S&P 500 Futures",    "color": "#00FF88"},
    "BTC": {"ticker": "BTC-USD", "name": "Bitcoin",            "color": "#F7931A"},
    "ETH": {"ticker": "ETH-USD", "name": "Ethereum",           "color": "#8B5CF6"},
}

VALUE_AREA_PCT = 0.70   # 70% do volume = Value Area
VWAP_STD_BANDS = [1, 2] # bandas de desvio padrão do VWAP

# ──────────────────────────────────────────────
# 1. RECOLHA DE DADOS
# ──────────────────────────────────────────────
def load_intraday(ticker: str, period: str = "2y",
                  interval: str = "1h") -> pd.DataFrame:
    df = yf.download(ticker, period=period,
                     interval=interval, progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                  for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["open","high","low","close","volume"]].dropna()
    return df

def load_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="5y", interval="1d", progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                  for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["open","high","low","close","volume"]].dropna()
    return df

# ──────────────────────────────────────────────
# 2. VWAP
# ──────────────────────────────────────────────
def calc_vwap(df: pd.DataFrame,
              reset_daily: bool = True) -> pd.DataFrame:
    """
    Calcula VWAP com reset diário (standard para day trading).
    Também calcula bandas de desvio padrão (+1σ, +2σ, -1σ, -2σ).

    VWAP = Σ(Typical Price × Volume) / Σ(Volume)
    Typical Price = (High + Low + Close) / 3
    """
    df = df.copy()
    df["tp"]     = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"] = df["tp"] * df["volume"]
    df["date"]   = df.index.date

    if reset_daily:
        df["cum_vol"]    = df.groupby("date")["volume"].cumsum()
        df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
    else:
        df["cum_vol"]    = df["volume"].cumsum()
        df["cum_tp_vol"] = df["tp_vol"].cumsum()

    df["vwap"] = df["cum_tp_vol"] / df["cum_vol"]

    # Desvio padrão para bandas
    df["vwap_var"] = (
        df.groupby("date").apply(
            lambda g: ((g["tp"] - g["vwap"]) ** 2 * g["volume"]).cumsum() / g["cum_vol"]
        ).reset_index(level=0, drop=True)
    )
    df["vwap_std"] = np.sqrt(df["vwap_var"].clip(lower=0))

    for b in VWAP_STD_BANDS:
        df[f"vwap_upper_{b}"] = df["vwap"] + b * df["vwap_std"]
        df[f"vwap_lower_{b}"] = df["vwap"] - b * df["vwap_std"]

    # Posição relativa ao VWAP
    df["above_vwap"] = df["close"] > df["vwap"]
    df["vwap_pct"]   = (df["close"] - df["vwap"]) / df["vwap"] * 100

    return df

def vwap_stats(df: pd.DataFrame) -> dict:
    """Estatísticas do VWAP: % do tempo acima/abaixo, mean reversion rate."""
    above = df["above_vwap"].mean() * 100
    below = 100 - above

    # Mean reversion: quantas vezes o preço voltou ao VWAP após afastar > 0.5%
    far_above = df["vwap_pct"] > 0.5
    far_below = df["vwap_pct"] < -0.5
    reverted_up   = (far_below.shift(1) & ~far_below).sum()
    reverted_down = (far_above.shift(1) & ~far_above).sum()
    total_far = far_above.sum() + far_below.sum()
    reversion_rate = (reverted_up + reverted_down) / max(total_far, 1) * 100

    return {
        "pct_above_vwap": above,
        "pct_below_vwap": below,
        "reversion_rate": reversion_rate,
        "mean_distance_pct": df["vwap_pct"].abs().mean(),
    }

# ──────────────────────────────────────────────
# 3. VOLUME PROFILE
# ──────────────────────────────────────────────
def calc_volume_profile(df: pd.DataFrame,
                        n_bins: int = 50) -> dict:
    """
    Calcula o Volume Profile para todo o período.

    Divide o range de preço em N bins e soma o volume de cada candle
    no bin correspondente ao seu typical price.

    POC = bin com maior volume
    Value Area = 70% do volume total, centrado no POC
    """
    price_min = df["low"].min()
    price_max = df["high"].max()
    bins      = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    vol_profile = np.zeros(n_bins)
    tp = (df["high"] + df["low"] + df["close"]) / 3

    for i in range(len(df)):
        idx = np.digitize(tp.iloc[i], bins) - 1
        idx = min(max(idx, 0), n_bins - 1)
        vol_profile[idx] += df["volume"].iloc[i]

    # POC
    poc_idx   = np.argmax(vol_profile)
    poc_price = bin_centers[poc_idx]

    # Value Area (70% do volume, expandindo do POC)
    target_vol = vol_profile.sum() * VALUE_AREA_PCT
    va_vol     = vol_profile[poc_idx]
    lo, hi     = poc_idx, poc_idx

    while va_vol < target_vol:
        up_vol   = vol_profile[hi + 1] if hi + 1 < n_bins else 0
        down_vol = vol_profile[lo - 1] if lo - 1 >= 0 else 0
        if up_vol >= down_vol and hi + 1 < n_bins:
            hi += 1
            va_vol += up_vol
        elif lo - 1 >= 0:
            lo -= 1
            va_vol += down_vol
        else:
            break

    vah = bin_centers[hi]
    val = bin_centers[lo]

    return {
        "profile":     vol_profile,
        "bins":        bin_centers,
        "poc":         poc_price,
        "vah":         vah,
        "val":         val,
        "price_min":   price_min,
        "price_max":   price_max,
        "poc_vol_pct": vol_profile[poc_idx] / vol_profile.sum() * 100,
    }

def calc_rolling_volume_profile(df: pd.DataFrame,
                                 window_days: int = 20,
                                 n_bins: int = 30) -> pd.DataFrame:
    """
    Volume Profile rolante — calcula POC, VAH, VAL
    para uma janela deslizante de N dias.
    """
    df = df.copy()
    df["date"] = df.index.date
    dates = sorted(df["date"].unique())

    poc_list, vah_list, val_list = [], [], []
    idx_list = []

    for i, d in enumerate(dates):
        if i < window_days:
            poc_list.append(np.nan)
            vah_list.append(np.nan)
            val_list.append(np.nan)
            idx_list.append(d)
            continue

        window = df[df["date"].isin(dates[i-window_days:i])]
        vp = calc_volume_profile(window, n_bins=n_bins)
        poc_list.append(vp["poc"])
        vah_list.append(vp["vah"])
        val_list.append(vp["val"])
        idx_list.append(d)

    result = pd.DataFrame({
        "date": idx_list,
        "poc":  poc_list,
        "vah":  vah_list,
        "val":  val_list,
    })
    result["date"] = pd.to_datetime(result["date"])
    result = result.set_index("date")
    return result

# ──────────────────────────────────────────────
# 4. CUMULATIVE DELTA
# ──────────────────────────────────────────────
def calc_cumulative_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimativa de Cumulative Delta sem dados de tape.

    Método: se candle bullish (close > open) → volume é buying pressure
            se candle bearish (close < open) → volume é selling pressure
            se doji → split 50/50

    Delta = buy_volume - sell_volume
    Cumulative Delta = soma acumulada do delta

    Divergência entre preço e cumulative delta = sinal de fraqueza.
    """
    df = df.copy()

    candle_range = (df["high"] - df["low"]).replace(0, 1e-10)
    buy_pct  = (df["close"] - df["low"]) / candle_range
    sell_pct = (df["high"] - df["close"]) / candle_range

    df["buy_vol"]  = df["volume"] * buy_pct
    df["sell_vol"] = df["volume"] * sell_pct
    df["delta"]    = df["buy_vol"] - df["sell_vol"]
    df["cum_delta"] = df["delta"].cumsum()

    # Normalizar para comparação visual
    df["cum_delta_norm"] = (
        (df["cum_delta"] - df["cum_delta"].min()) /
        (df["cum_delta"].max() - df["cum_delta"].min() + 1e-10)
    )
    df["close_norm"] = (
        (df["close"] - df["close"].min()) /
        (df["close"].max() - df["close"].min() + 1e-10)
    )

    # Divergência: delta e preço em direcções opostas (rolling 10)
    delta_dir = df["cum_delta"].diff(10).apply(np.sign)
    price_dir = df["close"].diff(10).apply(np.sign)
    df["divergence"] = (delta_dir != price_dir) & (delta_dir != 0)

    return df

def delta_stats(df: pd.DataFrame) -> dict:
    """Estatísticas do Cumulative Delta."""
    bull_sessions = (df["delta"] > 0).mean() * 100
    divergences   = df["divergence"].sum()
    total_bars    = len(df)
    div_rate      = divergences / total_bars * 100

    return {
        "pct_bullish_bars":  bull_sessions,
        "pct_bearish_bars":  100 - bull_sessions,
        "divergence_count":  divergences,
        "divergence_rate":   div_rate,
        "avg_delta_per_bar": df["delta"].mean(),
    }

# ──────────────────────────────────────────────
# 5. VISUALIZAÇÕES
# ──────────────────────────────────────────────
def plot_vwap_dashboard(df: pd.DataFrame, asset: str,
                        color: str, last_n_days: int = 30):
    """Dashboard VWAP com bandas — últimos N dias."""
    df = df.copy()
    cutoff = df.index[-1] - pd.Timedelta(days=last_n_days)
    df = df[df.index >= cutoff]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)

    # Preço + VWAP + Bandas
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df["close"], color=color, lw=1.5,
             label="Price", alpha=0.9)
    ax1.plot(df.index, df["vwap"],  color="white", lw=1.5,
             label="VWAP", linestyle="--")
    ax1.fill_between(df.index,
                     df["vwap_upper_1"], df["vwap_lower_1"],
                     alpha=0.1, color="white", label="±1σ")
    ax1.fill_between(df.index,
                     df["vwap_upper_2"], df["vwap_lower_2"],
                     alpha=0.05, color="white", label="±2σ")
    ax1.plot(df.index, df["vwap_upper_2"], color="#EF4444",
             lw=0.8, ls=":", alpha=0.6)
    ax1.plot(df.index, df["vwap_lower_2"], color="#22C55E",
             lw=0.8, ls=":", alpha=0.6)

    ax1.set_title(f"{asset} — VWAP with Standard Deviation Bands "
                  f"(Last {last_n_days} Days)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Price")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(axis="y", alpha=0.2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)
    sns.despine(ax=ax1)

    # Distance from VWAP %
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    cols = ["#22C55E" if v >= 0 else "#EF4444" for v in df["vwap_pct"]]
    ax2.bar(df.index, df["vwap_pct"], color=cols, alpha=0.7, width=0.03)
    ax2.axhline(0, color="white", lw=0.8)
    ax2.axhline(0.5,  color="#EF4444", lw=0.6, ls="--", alpha=0.5)
    ax2.axhline(-0.5, color="#22C55E", lw=0.6, ls="--", alpha=0.5)
    ax2.set_ylabel("% from VWAP")
    ax2.grid(axis="y", alpha=0.2)
    sns.despine(ax=ax2)

    # Volume
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    vcols = ["#22C55E" if c >= o else "#EF4444"
             for c, o in zip(df["close"], df["open"])]
    ax3.bar(df.index, df["volume"], color=vcols, alpha=0.7, width=0.03)
    ax3.set_ylabel("Volume")
    ax3.grid(axis="y", alpha=0.2)
    sns.despine(ax=ax3)

    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{asset}_vwap.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 {fname}")


def plot_volume_profile(df_daily: pd.DataFrame,
                        vp: dict, rolling_vp: pd.DataFrame,
                        asset: str, color: str):
    """Volume Profile: distribuição + POC/VAH/VAL histórico."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle(f"{asset} — Volume Profile Analysis",
                 fontsize=13, fontweight="bold")

    # 1. Volume Profile horizontal (últimos 6 meses)
    ax = axes[0]
    cutoff = df_daily.index[-1] - pd.Timedelta(days=180)
    df6m   = df_daily[df_daily.index >= cutoff]
    vp6m   = calc_volume_profile(df6m, n_bins=50)

    # Normalizar para visualização
    profile_norm = vp6m["profile"] / vp6m["profile"].max()
    bars = ax.barh(vp6m["bins"], profile_norm,
                   height=(vp6m["price_max"]-vp6m["price_min"])/50,
                   color=color, alpha=0.5)

    # Colorir VAH-VAL a verde
    for i, (b, p) in enumerate(zip(vp6m["bins"], profile_norm)):
        if vp6m["val"] <= b <= vp6m["vah"]:
            bars[i].set_color("#22C55E")
            bars[i].set_alpha(0.8)

    ax.axhline(vp6m["poc"], color="white",   lw=2,
               label=f"POC: {vp6m['poc']:,.1f}")
    ax.axhline(vp6m["vah"], color="#EF4444", lw=1.5, ls="--",
               label=f"VAH: {vp6m['vah']:,.1f}")
    ax.axhline(vp6m["val"], color="#22C55E", lw=1.5, ls="--",
               label=f"VAL: {vp6m['val']:,.1f}")

    ax.set_xlabel("Relative Volume")
    ax.set_ylabel("Price")
    ax.set_title("Volume Profile (Last 6 Months)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    sns.despine(ax=ax)

    # 2. Rolling POC/VAH/VAL histórico
    ax2 = axes[1]
    valid = rolling_vp.dropna()
    ax2.plot(df_daily.index, df_daily["close"],
             color=color, lw=1.5, alpha=0.8, label="Price")
    ax2.plot(valid.index, valid["poc"],
             color="white",   lw=1.5, ls="--", label="Rolling POC (20d)")
    ax2.plot(valid.index, valid["vah"],
             color="#EF4444", lw=1,   ls=":",  label="Rolling VAH")
    ax2.plot(valid.index, valid["val"],
             color="#22C55E", lw=1,   ls=":",  label="Rolling VAL")
    ax2.fill_between(valid.index, valid["val"], valid["vah"],
                     alpha=0.05, color="white")

    ax2.set_title("Rolling Value Area (20-day window)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Price")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)
    sns.despine(ax=ax2)

    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{asset}_volume_profile.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 {fname}")


def plot_cumulative_delta(df: pd.DataFrame, asset: str,
                          color: str, last_n_days: int = 60):
    """Cumulative Delta vs Price — detectar divergências."""
    df = df.copy()
    cutoff = df.index[-1] - pd.Timedelta(days=last_n_days)
    df = df[df.index >= cutoff]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                             gridspec_kw={"height_ratios": [3, 2, 1]},
                             sharex=True)
    fig.suptitle(f"{asset} — Cumulative Delta Analysis "
                 f"(Last {last_n_days} Days)",
                 fontsize=13, fontweight="bold")

    # Preço
    ax1 = axes[0]
    ax1.plot(df.index, df["close"], color=color, lw=2, label="Price")
    divs = df[df["divergence"]]
    ax1.scatter(divs.index, divs["close"], color="#FF6B35",
                marker="x", s=60, zorder=5,
                label=f"Divergence ({len(divs)})")
    ax1.set_ylabel("Price")
    ax1.set_title("Price", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.2)
    sns.despine(ax=ax1)

    # Cumulative Delta
    ax2 = axes[1]
    ax2.plot(df.index, df["cum_delta"], color="#F59E0B", lw=1.5,
             label="Cumulative Delta")
    ax2.fill_between(df.index, df["cum_delta"], 0,
                     where=df["cum_delta"] >= 0,
                     alpha=0.2, color="#22C55E")
    ax2.fill_between(df.index, df["cum_delta"], 0,
                     where=df["cum_delta"] < 0,
                     alpha=0.2, color="#EF4444")
    ax2.axhline(0, color="white", lw=0.8)
    ax2.set_ylabel("Cum. Delta")
    ax2.set_title("Cumulative Delta (Buy − Sell Volume)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.2)
    sns.despine(ax=ax2)

    # Bar Delta
    ax3 = axes[2]
    dcols = ["#22C55E" if d >= 0 else "#EF4444" for d in df["delta"]]
    ax3.bar(df.index, df["delta"], color=dcols, alpha=0.7, width=0.04)
    ax3.axhline(0, color="white", lw=0.8)
    ax3.set_ylabel("Bar Delta")
    ax3.grid(axis="y", alpha=0.2)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30)
    sns.despine(ax=ax3)

    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{asset}_cumulative_delta.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 {fname}")


def plot_combined_dashboard(df_intra: pd.DataFrame,
                             vp: dict,
                             asset: str, color: str,
                             last_n_days: int = 20):
    """Dashboard final: Price + VWAP + Volume Profile + Delta."""
    cutoff = df_intra.index[-1] - pd.Timedelta(days=last_n_days)
    df = df_intra[df_intra.index >= cutoff].copy()

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 2,
                            height_ratios=[3, 1.5, 1],
                            width_ratios=[3, 1],
                            hspace=0.15, wspace=0.05)

    # Preço + VWAP + POC/VAH/VAL
    ax_price = fig.add_subplot(gs[0, 0])
    ax_price.plot(df.index, df["close"], color=color, lw=1.5, label="Price")
    ax_price.plot(df.index, df["vwap"],  color="white", lw=1.2,
                  ls="--", label="VWAP")
    ax_price.fill_between(df.index,
                           df["vwap_upper_1"], df["vwap_lower_1"],
                           alpha=0.08, color="white")
    ax_price.axhline(vp["poc"], color="yellow", lw=1.2,
                     ls="--", alpha=0.8, label=f"POC {vp['poc']:,.0f}")
    ax_price.axhline(vp["vah"], color="#EF4444", lw=1,
                     ls=":", alpha=0.7, label=f"VAH {vp['vah']:,.0f}")
    ax_price.axhline(vp["val"], color="#22C55E", lw=1,
                     ls=":", alpha=0.7, label=f"VAL {vp['val']:,.0f}")
    ax_price.set_title(f"{asset} — Order Flow Dashboard (Last {last_n_days} Days)",
                       fontsize=12, fontweight="bold")
    ax_price.set_ylabel("Price")
    ax_price.legend(fontsize=8, loc="upper left", ncol=2)
    ax_price.grid(axis="y", alpha=0.2)
    sns.despine(ax=ax_price)

    # Volume Profile lateral
    ax_vp = fig.add_subplot(gs[0, 1], sharey=ax_price)
    vp_recent = calc_volume_profile(df, n_bins=40)
    bar_height = (vp_recent["price_max"] - vp_recent["price_min"]) / 40
    for i, (b, p) in enumerate(zip(vp_recent["bins"], vp_recent["profile"])):
        c = "#22C55E" if vp_recent["val"] <= b <= vp_recent["vah"] else color
        c = "yellow" if abs(b - vp_recent["poc"]) < bar_height else c
        ax_vp.barh(b, p, height=bar_height, color=c, alpha=0.7)
    ax_vp.set_title("Vol Profile", fontsize=9)
    ax_vp.set_xlabel("Volume")
    plt.setp(ax_vp.yaxis.get_majorticklabels(), visible=False)
    sns.despine(ax=ax_vp)

    # Cumulative Delta
    ax_delta = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_delta.plot(df.index, df["cum_delta"], color="#F59E0B", lw=1.5)
    ax_delta.fill_between(df.index, df["cum_delta"], 0,
                           where=df["cum_delta"] >= 0, alpha=0.2, color="#22C55E")
    ax_delta.fill_between(df.index, df["cum_delta"], 0,
                           where=df["cum_delta"] < 0, alpha=0.2, color="#EF4444")
    ax_delta.axhline(0, color="white", lw=0.8)
    divs = df[df["divergence"]]
    if len(divs):
        ax_delta.scatter(divs.index,
                         divs["cum_delta"],
                         color="#FF6B35", marker="x", s=50, zorder=5)
    ax_delta.set_ylabel("Cum. Delta")
    ax_delta.set_title("Cumulative Delta", fontsize=10)
    ax_delta.grid(axis="y", alpha=0.2)
    sns.despine(ax=ax_delta)

    # Volume bars
    ax_vol = fig.add_subplot(gs[2, 0], sharex=ax_price)
    vcols = ["#22C55E" if c >= o else "#EF4444"
             for c, o in zip(df["close"], df["open"])]
    ax_vol.bar(df.index, df["volume"], color=vcols, alpha=0.7, width=0.04)
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(axis="y", alpha=0.2)
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation=30)
    sns.despine(ax=ax_vol)

    plt.suptitle(f"{asset} — VWAP + Volume Profile + Cumulative Delta",
                 fontsize=13, fontweight="bold", y=1.01)
    fname = f"{OUTPUT_DIR}/{asset}_order_flow_dashboard.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 {fname}")


# ──────────────────────────────────────────────
# 6. RELATÓRIO FINAL
# ──────────────────────────────────────────────
def print_report(all_stats: dict):
    print("\n" + "="*65)
    print("  ORDER FLOW ANALYSIS — FINAL REPORT")
    print("  NQ + ES | VWAP + Volume Profile + Cumulative Delta")
    print("="*65)

    for asset, stats in all_stats.items():
        print(f"\n  {'─'*50}")
        print(f"  {asset} — {ASSETS[asset]['name']}")
        print(f"  {'─'*50}")

        v = stats.get("vwap_stats", {})
        print(f"\n  VWAP:")
        print(f"    Time above VWAP    : {v.get('pct_above_vwap',0):.1f}%")
        print(f"    Time below VWAP    : {v.get('pct_below_vwap',0):.1f}%")
        print(f"    Mean reversion rate: {v.get('reversion_rate',0):.1f}%")
        print(f"    Avg distance VWAP  : {v.get('mean_distance_pct',0):.3f}%")

        vp = stats.get("volume_profile", {})
        print(f"\n  Volume Profile (Full Period):")
        print(f"    POC : {vp.get('poc',0):>12,.2f}")
        print(f"    VAH : {vp.get('vah',0):>12,.2f}")
        print(f"    VAL : {vp.get('val',0):>12,.2f}")
        print(f"    POC volume concentration: {vp.get('poc_vol_pct',0):.1f}%")

        d = stats.get("delta_stats", {})
        print(f"\n  Cumulative Delta:")
        print(f"    Bullish bars : {d.get('pct_bullish_bars',0):.1f}%")
        print(f"    Bearish bars : {d.get('pct_bearish_bars',0):.1f}%")
        print(f"    Divergences  : {d.get('divergence_count',0)} "
              f"({d.get('divergence_rate',0):.1f}% of bars)")

    print("\n" + "="*65)
    print("💡 INTERPRETATION GUIDE:")
    print("   VWAP above 50% time = bullish market structure")
    print("   High reversion rate = mean-reverting market (range bound)")
    print("   POC = magnet price — expect consolidation near it")
    print("   Delta divergence = potential reversal signal")
    print("\n⚠️  For educational and portfolio purposes only.\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("="*65)
    print("  📊 Order Flow Analysis — NQ & ES Futures")
    print("  VWAP + Volume Profile + Cumulative Delta")
    print("="*65 + "\n")

    all_stats = {}

    for asset, cfg in ASSETS.items():
        print(f"{'─'*50}")
        print(f"  {asset} — {cfg['name']}")
        print(f"{'─'*50}")

        # 1. Dados
        print("  📡 A carregar dados intraday 1h...")
        try:
            df_intra = load_intraday(cfg["ticker"], period="2y", interval="1h")
            print(f"  ✅ {len(df_intra)} barras horárias")
        except Exception as e:
            print(f"  ❌ {e}"); continue

        print("  📡 A carregar dados diários 5 anos...")
        try:
            df_daily = load_daily(cfg["ticker"])
            print(f"  ✅ {len(df_daily)} dias")
        except Exception as e:
            print(f"  ❌ {e}"); continue

        # 2. VWAP
        print("  🔍 A calcular VWAP...")
        df_vwap = calc_vwap(df_intra, reset_daily=True)
        vs = vwap_stats(df_vwap)

        # 3. Volume Profile
        print("  🔍 A calcular Volume Profile...")
        vp_full    = calc_volume_profile(df_daily, n_bins=50)
        rolling_vp = calc_rolling_volume_profile(df_daily, window_days=20)

        # 4. Cumulative Delta
        print("  🔍 A calcular Cumulative Delta...")
        df_delta = calc_cumulative_delta(df_vwap)
        ds = delta_stats(df_delta)

        all_stats[asset] = {
            "vwap_stats":     vs,
            "volume_profile": vp_full,
            "delta_stats":    ds,
        }

        # 5. Gráficos
        print("  📈 A gerar gráficos...")
        plot_vwap_dashboard(df_vwap, asset, cfg["color"])
        plot_volume_profile(df_daily, vp_full, rolling_vp, asset, cfg["color"])
        plot_cumulative_delta(df_delta, asset, cfg["color"])
        plot_combined_dashboard(df_delta, vp_full, asset, cfg["color"])

    # 6. Relatório
    print_report(all_stats)
    print("✅ Análise completa!\n")
