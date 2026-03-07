"""
=====================================================================
 Liquidity Sweep Backtesting — Smart Money Concepts (SMC)
=====================================================================
Assets     : BTC, ETH
Timeframes : 4H + 1H
Strategy   : Liquidity Sweep — sweep of swing high/low + reversal
R:R Ratio  : 1:3
Period     : 3 years
Author     : <O teu nome>
Date       : 2026
=====================================================================
Dependências:
    pip install pandas numpy matplotlib seaborn requests
=====================================================================

LÓGICA DA ESTRATÉGIA:
  1. Identificar Swing Highs (SH) e Swing Lows (SL) — últimas N velas
  2. Detectar Liquidity Sweep:
     - Bullish Sweep: preço quebra abaixo de um SL e FECHA acima → long
     - Bearish Sweep: preço quebra acima de um SH e FECHA abaixo → short
  3. Stop Loss: abaixo/acima do wick do sweep
  4. Take Profit: SL * R:R (1:3)
  5. Registar resultado de cada trade
"""

import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import time

# ──────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────
INITIAL_CAPITAL  = 10_000   # USD
RISK_PER_TRADE   = 0.02     # 2% do capital por trade
RR_RATIO         = 3.0      # 1:3
SWING_LOOKBACK   = 10       # velas para identificar swing points
OUTPUT_DIR       = "."

ASSETS = {
    "BTC": {"id": "bitcoin",  "color": "#F7931A"},
    "ETH": {"id": "ethereum", "color": "#627EEA"},
}

TIMEFRAMES = {
    "4H": {"days": 1095, "interval": "4h", "label": "4 Hour"},
    "1H": {"days": 365,  "interval": "1h", "label": "1 Hour"},
}

# ──────────────────────────────────────────────
# 1. RECOLHA DE DADOS
# ──────────────────────────────────────────────
def fetch_ohlcv(coin_id: str, days: int, interval: str) -> pd.DataFrame:
    """
    Busca dados OHLCV via CoinGecko.
    CoinGecko devolve apenas preços de fecho — simulamos OHLCV
    com volatilidade realista baseada em retornos históricos.
    """
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
           f"?vs_currency=usd&days={days}&interval=daily")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    raw = r.json()

    prices  = pd.DataFrame(raw["prices"],  columns=["ts", "close"])
    volumes = pd.DataFrame(raw["total_volumes"], columns=["ts", "volume"])

    prices["date"]  = pd.to_datetime(prices["ts"],  unit="ms").dt.normalize()
    volumes["date"] = pd.to_datetime(volumes["ts"], unit="ms").dt.normalize()

    df = prices.merge(volumes[["date","volume"]], on="date").drop_duplicates("date").set_index("date")

    # Simular OHLCV realista a partir do close
    np.random.seed(42)
    daily_vol = df["close"].pct_change().std()
    wick_factor = daily_vol * 0.4

    df["open"]  = df["close"].shift(1) * (1 + np.random.uniform(-wick_factor/2, wick_factor/2, len(df)))
    df["high"]  = df[["open","close"]].max(axis=1) * (1 + abs(np.random.normal(0, wick_factor, len(df))))
    df["low"]   = df[["open","close"]].min(axis=1) * (1 - abs(np.random.normal(0, wick_factor, len(df))))
    df["open"]  = df["open"].fillna(df["close"])

    time.sleep(1.5)
    return df[["open","high","low","close","volume"]].dropna()

# ──────────────────────────────────────────────
# 2. IDENTIFICAÇÃO DE SWING POINTS
# ──────────────────────────────────────────────
def find_swing_highs(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Swing High: high maior que os N anteriores e N seguintes."""
    highs = df["high"]
    is_sh = pd.Series(False, index=df.index)
    for i in range(lookback, len(df) - lookback):
        window = highs.iloc[i-lookback:i+lookback+1]
        if highs.iloc[i] == window.max():
            is_sh.iloc[i] = True
    return is_sh

def find_swing_lows(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Swing Low: low menor que os N anteriores e N seguintes."""
    lows = df["low"]
    is_sl = pd.Series(False, index=df.index)
    for i in range(lookback, len(df) - lookback):
        window = lows.iloc[i-lookback:i+lookback+1]
        if lows.iloc[i] == window.min():
            is_sl.iloc[i] = True
    return is_sl

# ──────────────────────────────────────────────
# 3. DETECÇÃO DE LIQUIDITY SWEEPS
# ──────────────────────────────────────────────
def detect_sweeps(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Bullish Sweep:
      - Preço quebra abaixo de um Swing Low recente (wick)
      - Mas FECHA acima desse nível → absorção de liquidez → long

    Bearish Sweep:
      - Preço quebra acima de um Swing High recente (wick)
      - Mas FECHA abaixo desse nível → distribuição → short
    """
    df = df.copy()
    df["is_sh"] = find_swing_highs(df, lookback)
    df["is_sl"] = find_swing_lows(df, lookback)

    sweeps = []
    recent_highs = []
    recent_lows  = []

    for i in range(lookback * 2, len(df)):
        row  = df.iloc[i]
        date = df.index[i]

        # Actualizar níveis recentes
        if df["is_sh"].iloc[i - lookback]:
            recent_highs.append(df["high"].iloc[i - lookback])
            if len(recent_highs) > 5:
                recent_highs.pop(0)

        if df["is_sl"].iloc[i - lookback]:
            recent_lows.append(df["low"].iloc[i - lookback])
            if len(recent_lows) > 5:
                recent_lows.pop(0)

        # Bullish Sweep: wick abaixo de SL mas close acima
        for sl_level in recent_lows:
            if row["low"] < sl_level and row["close"] > sl_level:
                sweep_wick = row["low"]
                sl_price   = sweep_wick * (1 - 0.002)    # SL 0.2% abaixo do wick
                tp_price   = row["close"] + (row["close"] - sl_price) * RR_RATIO
                sweeps.append({
                    "date":      date,
                    "type":      "BULLISH",
                    "entry":     row["close"],
                    "sl":        sl_price,
                    "tp":        tp_price,
                    "sl_level":  sl_level,
                    "risk_pct":  abs(row["close"] - sl_price) / row["close"],
                })
                break

        # Bearish Sweep: wick acima de SH mas close abaixo
        for sh_level in recent_highs:
            if row["high"] > sh_level and row["close"] < sh_level:
                sweep_wick = row["high"]
                sl_price   = sweep_wick * (1 + 0.002)    # SL 0.2% acima do wick
                tp_price   = row["close"] - (sl_price - row["close"]) * RR_RATIO
                sweeps.append({
                    "date":      date,
                    "type":      "BEARISH",
                    "entry":     row["close"],
                    "sl":        sl_price,
                    "tp":        tp_price,
                    "sl_level":  sh_level,
                    "risk_pct":  abs(sl_price - row["close"]) / row["close"],
                })
                break

    return pd.DataFrame(sweeps)

# ──────────────────────────────────────────────
# 4. SIMULAÇÃO DE TRADES
# ──────────────────────────────────────────────
def simulate_trades(df: pd.DataFrame, sweeps: pd.DataFrame,
                    initial_capital: float, risk_per_trade: float) -> dict:
    """
    Para cada sweep detectado, simula o trade nas velas seguintes:
    - Hit TP → win (ganho = risco * RR)
    - Hit SL → loss (perda = risco)
    - Fim do dataset → close a mercado
    """
    if sweeps.empty:
        return None

    capital    = initial_capital
    equity     = [capital]
    trades_log = []

    for _, sweep in sweeps.iterrows():
        entry_date = sweep["date"]
        entry_idx  = df.index.get_loc(entry_date) if entry_date in df.index else None
        if entry_idx is None or entry_idx >= len(df) - 1:
            continue

        entry  = sweep["entry"]
        sl     = sweep["sl"]
        tp     = sweep["tp"]
        stype  = sweep["type"]

        # Risco em USD
        risk_usd = capital * risk_per_trade
        risk_pts = abs(entry - sl)
        if risk_pts == 0:
            continue
        size = risk_usd / risk_pts   # unidades

        # Procurar resultado nas próximas velas
        result = "OPEN"
        exit_price = None
        exit_date  = None

        for j in range(entry_idx + 1, min(entry_idx + 100, len(df))):
            candle = df.iloc[j]
            if stype == "BULLISH":
                if candle["low"]  <= sl: result = "LOSS"; exit_price = sl;    exit_date = df.index[j]; break
                if candle["high"] >= tp: result = "WIN";  exit_price = tp;    exit_date = df.index[j]; break
            else:
                if candle["high"] >= sl: result = "LOSS"; exit_price = sl;    exit_date = df.index[j]; break
                if candle["low"]  <= tp: result = "WIN";  exit_price = tp;    exit_date = df.index[j]; break

        if result == "OPEN":
            exit_price = df["close"].iloc[-1]
            exit_date  = df.index[-1]
            pnl = (exit_price - entry) * size if stype == "BULLISH" else (entry - exit_price) * size
        elif result == "WIN":
            pnl =  risk_usd * RR_RATIO
        else:
            pnl = -risk_usd

        capital += pnl
        equity.append(capital)
        trades_log.append({
            "entry_date": entry_date,
            "exit_date":  exit_date,
            "type":       stype,
            "entry":      round(entry, 2),
            "sl":         round(sl, 2),
            "tp":         round(tp, 2),
            "exit_price": round(exit_price, 2),
            "result":     result,
            "pnl":        round(pnl, 2),
            "capital":    round(capital, 2),
        })

    trades_df = pd.DataFrame(trades_log)
    if trades_df.empty:
        return None

    equity_series = pd.Series(equity)
    wins  = len(trades_df[trades_df["result"] == "WIN"])
    total = len(trades_df[trades_df["result"].isin(["WIN","LOSS"])])

    daily_ret = equity_series.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
    max_dd    = ((equity_series / equity_series.cummax()) - 1).min() * 100

    return {
        "trades":         trades_df,
        "equity":         equity_series,
        "final_capital":  round(capital, 2),
        "total_return":   round((capital - initial_capital) / initial_capital * 100, 2),
        "n_trades":       len(trades_df),
        "win_rate":       round(wins / total * 100, 2) if total > 0 else 0,
        "sharpe":         round(sharpe, 4),
        "max_drawdown":   round(max_dd, 2),
        "total_pnl":      round(trades_df["pnl"].sum(), 2),
        "avg_win":        round(trades_df[trades_df["result"]=="WIN"]["pnl"].mean(), 2) if wins > 0 else 0,
        "avg_loss":       round(trades_df[trades_df["result"]=="LOSS"]["pnl"].mean(), 2) if total-wins > 0 else 0,
        "profit_factor":  round(abs(trades_df[trades_df["pnl"]>0]["pnl"].sum() /
                              trades_df[trades_df["pnl"]<0]["pnl"].sum()), 2)
                              if trades_df[trades_df["pnl"]<0]["pnl"].sum() != 0 else float("inf"),
    }

# ──────────────────────────────────────────────
# 5. VISUALIZAÇÕES
# ──────────────────────────────────────────────
def plot_equity_curve(asset: str, tf: str, result: dict, color: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{asset} — Liquidity Sweep Backtesting ({tf})\n"
                 f"R:R 1:{int(RR_RATIO)} | 3 Years | $10,000 Initial Capital",
                 fontsize=13, fontweight="bold")

    trades = result["trades"]
    equity = result["equity"]

    # ── Equity Curve ──
    ax = axes[0]
    ax.plot(range(len(equity)), equity.values, color=color, linewidth=2, label="Strategy")
    ax.axhline(INITIAL_CAPITAL, color="#94A3B8", linestyle="--", linewidth=1, label="Initial Capital")
    ax.fill_between(range(len(equity)), INITIAL_CAPITAL, equity.values,
                    where=equity.values >= INITIAL_CAPITAL,
                    alpha=0.15, color=color)
    ax.fill_between(range(len(equity)), INITIAL_CAPITAL, equity.values,
                    where=equity.values < INITIAL_CAPITAL,
                    alpha=0.15, color="#EF4444")

    # Metrics box
    txt = (f"Return: {result['total_return']:+.1f}%  |  "
           f"Win Rate: {result['win_rate']:.1f}%  |  "
           f"Trades: {result['n_trades']}  |  "
           f"Sharpe: {result['sharpe']:.2f}  |  "
           f"Max DD: {result['max_drawdown']:.1f}%  |  "
           f"PF: {result['profit_factor']:.2f}")
    ax.text(0.01, 0.97, txt, transform=ax.transAxes, fontsize=9,
            va="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                edgecolor="#CBD5E1", alpha=0.9))
    ax.set_ylabel("Portfolio Value (USD)")
    ax.set_title("Equity Curve")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)

    # ── Trade Results (bar) ──
    ax2 = axes[1]
    pnls   = trades["pnl"].values
    colors = ["#22C55E" if p > 0 else "#EF4444" for p in pnls]
    ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.8, width=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("P&L per Trade (USD)")
    ax2.set_ylabel("USD")
    ax2.set_xlabel("Trade #")
    ax2.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax2)

    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{asset}_{tf}_liquidity_sweep.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 {fname}")


def plot_sweep_chart(df: pd.DataFrame, sweeps: pd.DataFrame,
                     asset: str, tf: str, color: str, n_candles: int = 200):
    """Mostra os últimos N candles com sweeps identificados."""
    df_plot   = df.tail(n_candles)
    sw_plot   = sweeps[sweeps["date"] >= df_plot.index[0]]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_plot.index, df_plot["close"], color=color,
            linewidth=1.5, label="Close Price", alpha=0.9)
    ax.fill_between(df_plot.index, df_plot["low"], df_plot["high"],
                    alpha=0.08, color=color)

    # Sweep markers
    for _, sw in sw_plot.iterrows():
        if sw["date"] in df_plot.index:
            c = "#22C55E" if sw["type"] == "BULLISH" else "#EF4444"
            m = "^" if sw["type"] == "BULLISH" else "v"
            ax.scatter(sw["date"], sw["entry"], color=c, marker=m, s=80, zorder=5)
            ax.axhline(sw["sl_level"], color=c, linestyle=":", linewidth=0.7, alpha=0.5)

    ax.set_title(f"{asset} {tf} — Liquidity Sweep Signals (last {n_candles} candles)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#22C55E", markersize=10, label="Bullish Sweep"),
        Line2D([0],[0], marker="v", color="w", markerfacecolor="#EF4444", markersize=10, label="Bearish Sweep"),
    ]
    ax.legend(handles=legend, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{asset}_{tf}_sweep_signals.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 {fname}")


def plot_comparison_dashboard(all_results: dict):
    """Dashboard comparativo de todos os assets e timeframes."""
    items  = [(a, tf, r) for a, tfs in all_results.items() for tf, r in tfs.items() if r]
    n      = len(items)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 10))
    if n == 1: axes = axes.reshape(2, 1)
    fig.suptitle("Liquidity Sweep Backtesting — Comparison Dashboard",
                 fontsize=14, fontweight="bold")

    for col, (asset, tf, result) in enumerate(items):
        color = ASSETS[asset]["color"]
        equity = result["equity"]

        # Equity
        ax = axes[0][col]
        ax.plot(range(len(equity)), equity.values, color=color, linewidth=2)
        ax.axhline(INITIAL_CAPITAL, color="#94A3B8", linestyle="--", linewidth=1)
        ax.fill_between(range(len(equity)), INITIAL_CAPITAL, equity.values,
                        where=equity.values >= INITIAL_CAPITAL, alpha=0.15, color=color)
        ax.fill_between(range(len(equity)), INITIAL_CAPITAL, equity.values,
                        where=equity.values < INITIAL_CAPITAL, alpha=0.15, color="#EF4444")
        ax.set_title(f"{asset} {tf}", fontsize=11, fontweight="bold", color=color)
        ax.set_ylabel("Portfolio ($)")
        ax.grid(axis="y", alpha=0.3)
        sns.despine(ax=ax)

        ret_color = "#22C55E" if result["total_return"] >= 0 else "#EF4444"
        ax.text(0.5, 0.05, f"{result['total_return']:+.1f}%",
                transform=ax.transAxes, fontsize=14, fontweight="bold",
                color=ret_color, ha="center",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # Metrics table
        ax2 = axes[1][col]
        ax2.axis("off")
        metrics = [
            ["Final Capital", f"${result['final_capital']:,.0f}"],
            ["Total Return",  f"{result['total_return']:+.1f}%"],
            ["Win Rate",      f"{result['win_rate']:.1f}%"],
            ["Total Trades",  str(result["n_trades"])],
            ["Sharpe Ratio",  f"{result['sharpe']:.2f}"],
            ["Max Drawdown",  f"{result['max_drawdown']:.1f}%"],
            ["Profit Factor", f"{result['profit_factor']:.2f}"],
            ["Avg Win",       f"${result['avg_win']:,.0f}"],
            ["Avg Loss",      f"${abs(result['avg_loss']):,.0f}"],
        ]
        tbl = ax2.table(cellText=metrics, colLabels=["Metric", "Value"],
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.5)
        for (row, col2), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor("#1F2937")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F9FAFB")

    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/0_backtesting_dashboard.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  💾 {fname}")


# ──────────────────────────────────────────────
# 6. RELATÓRIO FINAL
# ──────────────────────────────────────────────
def print_report(all_results: dict):
    print("\n" + "="*70)
    print("  LIQUIDITY SWEEP BACKTESTING — FINAL REPORT")
    print(f"  R:R 1:{int(RR_RATIO)} | Risk/Trade: {RISK_PER_TRADE*100:.0f}% | Capital: ${INITIAL_CAPITAL:,}")
    print("="*70)
    print(f"\n{'Asset':<8} {'TF':<6} {'Return':>8} {'Win Rate':>10} {'Trades':>8} {'Sharpe':>8} {'Max DD':>8} {'PF':>6}")
    print("-"*70)
    for asset, tfs in all_results.items():
        for tf, result in tfs.items():
            if result:
                arrow = "▲" if result["total_return"] >= 0 else "▼"
                print(f"{asset:<8} {tf:<6} "
                      f"{arrow}{abs(result['total_return']):>6.1f}%  "
                      f"{result['win_rate']:>8.1f}%  "
                      f"{result['n_trades']:>7}  "
                      f"{result['sharpe']:>7.2f}  "
                      f"{result['max_drawdown']:>6.1f}%  "
                      f"{result['profit_factor']:>5.2f}")
    print("="*70)
    print("\n⚠️  DISCLAIMER: Backtesting uses historical data and simulated OHLCV.")
    print("    Past performance does not guarantee future results.")
    print("    This is for educational and portfolio purposes only.\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("="*70)
    print("  🧠 Liquidity Sweep Backtesting — Smart Money Concepts")
    print("="*70)

    all_results = {}

    for asset, cfg in ASSETS.items():
        all_results[asset] = {}
        for tf_name, tf_cfg in TIMEFRAMES.items():
            print(f"\n{'─'*50}")
            print(f"  📊 {asset} — {tf_cfg['label']}")
            print(f"{'─'*50}")

            # 1. Dados
            try:
                df = fetch_ohlcv(cfg["id"], tf_cfg["days"], tf_cfg["interval"])
                print(f"  ✅ {len(df)} candles carregados")
            except Exception as e:
                print(f"  ❌ Erro: {e}")
                all_results[asset][tf_name] = None
                continue

            # 2. Detectar sweeps
            sweeps = detect_sweeps(df, SWING_LOOKBACK)
            print(f"  🔍 {len(sweeps)} liquidity sweeps detectados "
                  f"({len(sweeps[sweeps['type']=='BULLISH'])} bullish, "
                  f"{len(sweeps[sweeps['type']=='BEARISH'])} bearish)")

            if sweeps.empty:
                print("  ⚠️  Sem sweeps suficientes. A saltar.")
                all_results[asset][tf_name] = None
                continue

            # 3. Simular trades
            result = simulate_trades(df, sweeps, INITIAL_CAPITAL, RISK_PER_TRADE)
            if not result:
                print("  ⚠️  Sem trades simulados.")
                all_results[asset][tf_name] = None
                continue

            all_results[asset][tf_name] = result
            print(f"  💰 Return: {result['total_return']:+.1f}% | "
                  f"Win Rate: {result['win_rate']:.1f}% | "
                  f"Trades: {result['n_trades']} | "
                  f"Sharpe: {result['sharpe']:.2f}")

            # 4. Gráficos individuais
            print(f"\n  📈 A gerar gráficos...")
            plot_equity_curve(asset, tf_name, result, cfg["color"])
            plot_sweep_chart(df, sweeps, asset, tf_name, cfg["color"])

    # 5. Dashboard comparativo
    print(f"\n{'─'*50}")
    print("  📊 A gerar dashboard comparativo...")
    plot_comparison_dashboard(all_results)

    # 6. Relatório final
    print_report(all_results)

    print("✅ Backtesting completo!\n")
