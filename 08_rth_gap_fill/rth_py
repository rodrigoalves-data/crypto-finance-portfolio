import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta

OUTPUT_DIR = "."

# ──────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────
ASSETS = {
    "NQ": {"ticker": "NQ=F", "name": "Nasdaq 100 Futures", "color": "#627EEA"},
    "ES": {"ticker": "ES=F", "name": "S&P 500 Futures",    "color": "#00FF88"},
}

FILL_LEVELS = [0.25, 0.50, 0.75, 1.00]

# ──────────────────────────────────────────────
# 1. RECOLHA DE DADOS
# ──────────────────────────────────────────────
def load_daily(ticker: str) -> pd.DataFrame:
    """Dados diários — para identificar gaps e fills no mesmo dia."""
    df = yf.download(ticker, period="5y", interval="1d", progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                  for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["open","high","low","close","volume"]].dropna()
    return df

def load_intraday(ticker: str) -> pd.DataFrame:
    """
    Dados intraday 1h — para analisar fill na primeira hora.
    Yahoo Finance tem máx 730 dias em 1h, por isso usamos 2 anos.
    """
    df = yf.download(ticker, period="2y", interval="1h", progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                  for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["open","high","low","close","volume"]].dropna()
    return df

# ──────────────────────────────────────────────
# 2. IDENTIFICAR RTH GAPS
# ──────────────────────────────────────────────
def identify_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica todos os RTH gaps:
    - Gap = open do dia - close do dia anterior
    - Gap % = gap / close anterior * 100
    - Bullish = open > prev_close
    - Bearish = open < prev_close
    """
    df = df.copy()
    df["prev_close"] = df["close"].shift(1)
    df["gap"]        = df["open"] - df["prev_close"]
    df["gap_pct"]    = df["gap"] / df["prev_close"] * 100
    df["gap_type"]   = np.where(df["gap"] > 0, "BULLISH",
                        np.where(df["gap"] < 0, "BEARISH", "NONE"))

    # Remover dias sem gap
    gaps = df[df["gap_type"] != "NONE"].copy()
    gaps = gaps.dropna(subset=["prev_close"])
    return gaps

# ──────────────────────────────────────────────
# 3. CALCULAR FILL STATISTICS
# ──────────────────────────────────────────────
def calc_fill_stats(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada gap, verifica se foi preenchido a 25/50/75/100%
    usando o high/low do mesmo dia.

    Bullish gap fill: preço desce de volta ao prev_close
      - O low do dia toca o nível de fill
    Bearish gap fill: preço sobe de volta ao prev_close
      - O high do dia toca o nível de fill
    """
    gaps = identify_gaps(df_daily)
    results = []

    for date, row in gaps.iterrows():
        gap_type   = row["gap_type"]
        prev_close = row["prev_close"]
        open_price = row["open"]
        high       = row["high"]
        low        = row["low"]
        gap_size   = abs(row["gap"])

        record = {
            "date":       date,
            "type":       gap_type,
            "prev_close": prev_close,
            "open":       open_price,
            "gap":        row["gap"],
            "gap_pct":    row["gap_pct"],
            "gap_size":   gap_size,
        }

        for level in FILL_LEVELS:
            if gap_type == "BULLISH":
                # Fill level = open - (gap * level) → descida
                fill_price = open_price - (gap_size * level)
                filled = low <= fill_price
            else:
                # Fill level = open + (gap_size * level) → subida
                fill_price = open_price + (gap_size * level)
                filled = high >= fill_price

            record[f"fill_{int(level*100)}"] = filled
            record[f"fill_{int(level*100)}_price"] = fill_price

        results.append(record)

    return pd.DataFrame(results)

def calc_days_to_fill(df_daily: pd.DataFrame,
                      gaps_df: pd.DataFrame,
                      level: float = 1.0,
                      max_days: int = 20) -> pd.Series:
    """
    Para gaps que não fecharam no mesmo dia,
    calcula quantos dias demoraram a fechar (máx N dias).
    """
    days_list = []
    col = f"fill_{int(level*100)}"

    for _, gap in gaps_df.iterrows():
        if gap[col]:
            days_list.append(0)  # fechou no mesmo dia
            continue

        gap_type   = gap["type"]
        prev_close = gap["prev_close"]
        open_price = gap["open"]
        gap_size   = gap["gap_size"]
        fill_price = (open_price - gap_size * level if gap_type == "BULLISH"
                      else open_price + gap_size * level)

        # Procurar nos dias seguintes
        future = df_daily.loc[df_daily.index > pd.Timestamp(gap["date"])].head(max_days)
        filled_day = None
        for i, (fdate, frow) in enumerate(future.iterrows()):
            if gap_type == "BULLISH" and frow["low"] <= fill_price:
                filled_day = i + 1
                break
            elif gap_type == "BEARISH" and frow["high"] >= fill_price:
                filled_day = i + 1
                break

        days_list.append(filled_day if filled_day else np.nan)

    return pd.Series(days_list)

def calc_first_hour_fill(df_intraday: pd.DataFrame,
                         gaps_df: pd.DataFrame,
                         level: float = 0.50) -> pd.Series:
    """
    Verifica se o gap foi preenchido a X% na primeira hora de trading
    (09:30 - 10:30 ET).
    """
    results = []
    col = f"fill_{int(level*100)}"

    for _, gap in gaps_df.iterrows():
        date = pd.Timestamp(gap["date"]).date()
        gap_type   = gap["type"]
        open_price = gap["open"]
        gap_size   = gap["gap_size"]
        fill_price = (open_price - gap_size * level if gap_type == "BULLISH"
                      else open_price + gap_size * level)

        # Filtrar primeira hora
        first_hour = df_intraday[
            (df_intraday.index.date == date) &
            (df_intraday.index.hour < 11)    # 09:30-10:30
        ]

        if first_hour.empty:
            results.append(np.nan)
            continue

        if gap_type == "BULLISH":
            filled = (first_hour["low"] <= fill_price).any()
        else:
            filled = (first_hour["high"] >= fill_price).any()

        results.append(filled)

    return pd.Series(results)

# ──────────────────────────────────────────────
# 4. ANÁLISE COMPLETA
# ──────────────────────────────────────────────
def full_analysis(asset: str, df_daily: pd.DataFrame,
                  df_intraday: pd.DataFrame) -> dict:

    gaps = calc_fill_stats(df_daily)
    bullish = gaps[gaps["type"] == "BULLISH"]
    bearish = gaps[gaps["type"] == "BEARISH"]

    results = {
        "total_gaps":    len(gaps),
        "bullish_gaps":  len(bullish),
        "bearish_gaps":  len(bearish),
        "avg_gap_pct":   gaps["gap_pct"].abs().mean(),
        "bullish": {},
        "bearish": {},
    }

    for gtype, gdf in [("bullish", bullish), ("bearish", bearish)]:
        if gdf.empty:
            continue

        # Fill rates
        for level in FILL_LEVELS:
            col = f"fill_{int(level*100)}"
            rate = gdf[col].mean() * 100
            results[gtype][f"fill_{int(level*100)}_rate"] = rate

        # Days to fill 100%
        days = calc_days_to_fill(df_daily, gdf, level=1.0)
        results[gtype]["days_to_fill_mean"] = days.mean()
        results[gtype]["days_to_fill_median"] = days.median()
        results[gtype]["same_day_fill_100"] = (days == 0).sum()

        # First hour fill 50%
        fh = calc_first_hour_fill(df_intraday, gdf, level=0.50)
        results[gtype]["first_hour_50_rate"] = fh.mean() * 100

        # Gap size distribution
        results[gtype]["avg_gap_pct"] = gdf["gap_pct"].abs().mean()
        results[gtype]["max_gap_pct"] = gdf["gap_pct"].abs().max()
        results[gtype]["gaps_df"]     = gdf

    return results

# ──────────────────────────────────────────────
# 5. VISUALIZAÇÕES
# ──────────────────────────────────────────────
def plot_fill_rates(results: dict, asset: str, color: str):
    bull = results["bullish"]
    bear = results["bearish"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{asset} — RTH Gap Fill Statistics (5 Years)",
                 fontsize=14, fontweight="bold")

    # 1. Fill rates por nível
    ax = axes[0, 0]
    levels     = ["25%", "50%", "75%", "100%"]
    keys       = ["fill_25_rate","fill_50_rate","fill_75_rate","fill_100_rate"]
    bull_rates = [bull.get(k, 0) for k in keys]
    bear_rates = [bear.get(k, 0) for k in keys]
    x = np.arange(len(levels))
    ax.bar(x - 0.2, bull_rates, 0.4, label="Bullish Gap",
           color="#22C55E", alpha=0.85)
    ax.bar(x + 0.2, bear_rates, 0.4, label="Bearish Gap",
           color="#EF4444", alpha=0.85)
    for i, (b, r) in enumerate(zip(bull_rates, bear_rates)):
        ax.text(i-0.2, b+0.5, f"{b:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax.text(i+0.2, r+0.5, f"{r:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Fill Rate (%)")
    ax.set_title("Gap Fill Rate by Level (Same Day)", fontsize=11, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)

    # 2. First hour 50% fill
    ax2 = axes[0, 1]
    fh_bull = bull.get("first_hour_50_rate", 0)
    fh_bear = bear.get("first_hour_50_rate", 0)
    bars = ax2.bar(["Bullish Gap\n(Fill 50%)", "Bearish Gap\n(Fill 50%)"],
                   [fh_bull, fh_bear],
                   color=["#22C55E", "#EF4444"], alpha=0.85, width=0.5)
    for bar, val in zip(bars, [fh_bull, fh_bear]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax2.axhline(50, color="white", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Fill Rate (%)")
    ax2.set_title("50% Fill in First Hour (09:30-10:30)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax2)

    # 3. Gap size distribution
    ax3 = axes[1, 0]
    bull_gaps = results["bullish"].get("gaps_df", pd.DataFrame())
    bear_gaps = results["bearish"].get("gaps_df", pd.DataFrame())
    if not bull_gaps.empty:
        ax3.hist(bull_gaps["gap_pct"], bins=30, color="#22C55E",
                 alpha=0.6, label="Bullish", density=True)
    if not bear_gaps.empty:
        ax3.hist(bear_gaps["gap_pct"].abs(), bins=30, color="#EF4444",
                 alpha=0.6, label="Bearish (abs)", density=True)
    ax3.set_xlabel("Gap Size (%)")
    ax3.set_ylabel("Density")
    ax3.set_title("Gap Size Distribution", fontsize=11, fontweight="bold")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax3)

    # 4. Days to fill 100%
    ax4 = axes[1, 1]

    # Remove the redundant call to calc_days_to_fill which was causing the TypeError
    # The values bull_days and bear_days are correctly fetched directly from the results dict below
    # for gtype, gdf, col in [("Bullish", bull_gaps, "#22C55E"),
    #                           ("Bearish", bear_gaps, "#EF4444")]:
    #     if gdf.empty: continue
    #     days = calc_days_to_fill(
    #         pd.DataFrame(), gdf, level=1.0)  # já calculado
    #     # Usar os dados já calculados
    #     bull_days = bull.get("days_to_fill_mean", np.nan)
    #     bear_days = bear.get("days_to_fill_mean", np.nan)

    days_data = {
        "Bullish": bull.get("days_to_fill_mean", 0),
        "Bearish": bear.get("days_to_fill_mean", 0),
    }
    bars2 = ax4.bar(days_data.keys(), days_data.values(),
                    color=["#22C55E","#EF4444"], alpha=0.85, width=0.4)
    for bar, val in zip(bars2, days_data.values()):
        if not np.isnan(val):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.1f} days", ha="center", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Average Days")
    ax4.set_title("Avg Days to Fill 100% of Gap", fontsize=11, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax4)

    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{asset}_gap_fill.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 {fname}")


def plot_gap_history(df_daily: pd.DataFrame, gaps_df: pd.DataFrame,
                     asset: str, color: str):
    """Mostra histórico de gaps com cores por tipo."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{asset} — RTH Gap History (5 Years)",
                 fontsize=13, fontweight="bold")

    # Preço
    ax = axes[0]
    ax.plot(df_daily.index, df_daily["close"],
            color=color, linewidth=1.5, alpha=0.9)

    # Marcar gaps
    bull = gaps_df[gaps_df["gap_type"] == "BULLISH"]
    bear = gaps_df[gaps_df["gap_type"] == "BEARISH"]
    ax.scatter(bull.index, bull["open"], color="#22C55E",
               marker="^", s=30, zorder=5, alpha=0.7, label="Bullish Gap")
    ax.scatter(bear.index, bear["open"], color="#EF4444",
               marker="v", s=30, zorder=5, alpha=0.7, label="Bearish Gap")

    ax.set_title("Price with Gap Markers", fontsize=11)
    ax.set_ylabel("Price")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)

    # Gap size ao longo do tempo
    ax2 = axes[1]
    ax2.bar(bull["date"], bull["gap_pct"],
            color="#22C55E", alpha=0.7, width=3, label="Bullish")
    ax2.bar(bear["date"], bear["gap_pct"],
            color="#EF4444", alpha=0.7, width=3, label="Bearish")
    ax2.axhline(0, color="white", linewidth=0.8)
    ax2.set_title("Gap Size (%)" + " Over Time", fontsize=11)
    ax2.set_ylabel("Gap %")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax2)

    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{asset}_gap_history.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 {fname}")


def plot_comparison(all_results: dict):
    """Dashboard comparativo NQ vs ES."""
    assets  = list(all_results.keys())
    metrics = {
        "fill_50_rate":        "Same Day 50% Fill (%)",
        "fill_100_rate":       "Same Day 100% Fill (%)",
        "first_hour_50_rate":  "First Hour 50% Fill (%)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("NQ vs ES — RTH Gap Fill Comparison",
                 fontsize=14, fontweight="bold")

    for ax, (metric, label) in zip(axes, metrics.items()):
        bull_vals = [all_results[a]["bullish"].get(metric, 0) for a in assets]
        bear_vals = [all_results[a]["bearish"].get(metric, 0) for a in assets]

        x = np.arange(len(assets))
        ax.bar(x - 0.2, bull_vals, 0.4, label="Bullish",
               color="#22C55E", alpha=0.85)
        ax.bar(x + 0.2, bear_vals, 0.4, label="Bearish",
               color="#EF4444", alpha=0.85)

        for i, (b, r) in enumerate(zip(bull_vals, bear_vals)):
            ax.text(i-0.2, b+0.5, f"{b:.1f}%", ha="center",
                    fontsize=9, fontweight="bold")
            ax.text(i+0.2, r+0.5, f"{r:.1f}%", ha="center",
                    fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(assets, fontsize=12)
        ax.set_ylabel("%")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/0_nq_es_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 {fname}")


# ──────────────────────────────────────────────
# 6. RELATÓRIO FINAL
# ──────────────────────────────────────────────
def print_report(all_results: dict):
    print("\n" + "="*65)
    print("  RTH GAP FILL STATISTICS — FINAL REPORT")
    print("  NQ (Nasdaq 100) + ES (S&P 500) | 5 Years")
    print("="*65)

    for asset, res in all_results.items():
        print(f"\n  {'─'*50}")
        print(f"  {asset} — {ASSETS[asset]['name']}")
        print(f"  {'─'*50}")
        print(f"  Total Gaps    : {res['total_gaps']}")
        print(f"  Bullish Gaps  : {res['bullish_gaps']} "
              f"({res['bullish_gaps']/res['total_gaps']*100:.1f}%)")
        print(f"  Bearish Gaps  : {res['bearish_gaps']} "
              f"({res['bearish_gaps']/res['total_gaps']*100:.1f}%)")
        print(f"  Avg Gap Size  : {res['avg_gap_pct']:.3f}%")

        for gtype in ["bullish", "bearish"]:
            g = res[gtype]
            if not g: continue
            arrow = "▲" if gtype == "bullish" else "▼"
            print(f"\n  {arrow} {gtype.upper()} GAPS:")
            print(f"    Fill 25%  same day : {g.get('fill_25_rate',0):.1f}%")
            print(f"    Fill 50%  same day : {g.get('fill_50_rate',0):.1f}%")
            print(f"    Fill 75%  same day : {g.get('fill_75_rate',0):.1f}%")
            print(f"    Fill 100% same day : {g.get('fill_100_rate',0):.1f}%")
            print(f"    Fill 50%  1st hour : {g.get('first_hour_50_rate',0):.1f}%")
            print(f"    Avg days to full fill: {g.get('days_to_fill_mean', np.nan):.1f}")

    print("\n" + "="*65)
    print("💡 EDGE INTERPRETATION:")
    print("   Fill rate > 70% = strong statistical edge")
    print("   Fill rate > 50% = moderate edge worth tracking")
    print("   First hour fill > 40% = intraday trading opportunity")
    print("\n⚠️  Past statistics do not guarantee future performance.")
    print("   For educational and portfolio purposes only.\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("="*65)
    print("  📊 RTH Gap Fill Statistics — NQ & ES Futures")
    print("="*65 + "\n")

    all_results = {}

    for asset, cfg in ASSETS.items():
        print(f"{'─'*50}")
        print(f"  {asset} — {cfg['name']}")
        print(f"{'─'*50}")

        # 1. Dados diários (5 anos)
        print("  📡 A carregar dados diários (5 anos)...")
        try:
            df_daily = load_daily(cfg["ticker"])
            print(f"  ✅ {len(df_daily)} dias | "
                  f"{df_daily.index[0].date()} → {df_daily.index[-1].date()}")
        except Exception as e:
            print(f"  ❌ {e}"); continue

        # 2. Dados intraday (2 anos — limite Yahoo Finance)
        print("  📡 A carregar dados intraday 1h (2 anos)...")
        try:
            df_intra = load_intraday(cfg["ticker"])
            print(f"  ✅ {len(df_intra)} barras horárias")
        except Exception as e:
            print(f"  ⚠️  Intraday error: {e}")
            df_intra = pd.DataFrame()

        # 3. Análise
        print("  🔍 A analisar gaps...")
        res = full_analysis(asset, df_daily, df_intra)
        all_results[asset] = res

        print(f"  ✅ {res['total_gaps']} gaps identificados "
              f"({res['bullish_gaps']} bullish | {res['bearish_gaps']} bearish)")

        # 4. Gráficos
        print("  📈 A gerar gráficos...")
        gaps_df = identify_gaps(df_daily)
        gaps_df = gaps_df.copy()
        gaps_df["date"] = pd.to_datetime(gaps_df.index)
        plot_fill_rates(res, asset, cfg["color"])
        plot_gap_history(df_daily, gaps_df, asset, cfg["color"])

    # 5. Comparação
    if len(all_results) == 2:
        print("\n  📊 A gerar comparaçãó NQ vs ES...")
        plot_comparison(all_results)

    # 6. Relatório
    print_report(all_results)

    print("✅ Análise completa!\n")
