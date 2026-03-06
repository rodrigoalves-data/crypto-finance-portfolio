"""
=====================================================================
 Portfolio Optimizer — Markowitz Modern Portfolio Theory
=====================================================================
Assets  : BTC, ETH, S&P 500, Gold, Silver, Apple, Palantir, US Treasuries
Goals   : Max Sharpe | Min Risk | Max Return
Period  : 2 years
Method  : Monte Carlo Simulation + Efficient Frontier
Author  : <O teu nome>
Date    : 2026
=====================================================================
Dependências:
    pip install pandas numpy matplotlib seaborn scipy yfinance requests
=====================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime, timedelta
import time

# ──────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────
INITIAL_CAPITAL  = 10_000
RISK_FREE_RATE   = 0.05      # 5% — US Treasury rate aprox.
N_PORTFOLIOS     = 10_000    # Monte Carlo simulations
PERIOD           = "2y"
OUTPUT_DIR       = "."

ASSETS = {
    "BTC":        {"type": "crypto",  "id": "bitcoin",  "color": "#F7931A"},
    "ETH":        {"type": "crypto",  "id": "ethereum", "color": "#627EEA"},
    "S&P 500":    {"type": "yahoo",   "id": "^GSPC",    "color": "#00FF88"},
    "Gold":       {"type": "yahoo",   "id": "GC=F",     "color": "#FFD93D"},
    "Silver":     {"type": "yahoo",   "id": "SI=F",     "color": "#C0C0C0"},
    "Apple":      {"type": "yahoo",   "id": "AAPL",     "color": "#A8B5C1"},
    "Palantir":   {"type": "yahoo",   "id": "PLTR",     "color": "#7C3AED"},
    "US Treasuries": {"type": "yahoo","id": "TLT",      "color": "#2D6A4F"},
}

# ──────────────────────────────────────────────
# 1. RECOLHA DE DADOS
# ──────────────────────────────────────────────
def fetch_crypto(coin_id: str, days: int = 730) -> pd.Series:
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
           f"?vs_currency=usd&days={days}&interval=daily")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["prices"], columns=["ts", "price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    s = df.drop_duplicates("date").set_index("date")["price"]
    time.sleep(1.5)
    return s

def fetch_yahoo(ticker: str) -> pd.Series:
    t    = yf.Ticker(ticker)
    hist = t.history(period=PERIOD, interval="1d")
    s    = hist["Close"].squeeze()
    s.index = pd.to_datetime(s.index).normalize()
    return s

def load_prices() -> pd.DataFrame:
    print("📡 A recolher dados...\n")
    series = {}
    for name, cfg in ASSETS.items():
        try:
            if cfg["type"] == "crypto":
                s = fetch_crypto(cfg["id"])
            else:
                s = fetch_yahoo(cfg["id"])
            # Normalizar timezone — remover tz de todos os índices
            s.index = pd.to_datetime(s.index).tz_localize(None)
            s.index = s.index.normalize()
            s.name = name
            series[name] = s
            print(f"  ✅ {name} — {len(s)} dias")
        except Exception as e:
            print(f"  ❌ {name} — {e}")

    df = pd.concat(series.values(), axis=1)
    df = df.ffill().bfill().dropna(how="all")
    print(f"\n  📅 Período: {df.index.min().date()} → {df.index.max().date()}")
    print(f"  📊 Activos: {df.shape[1]} | Dias: {df.shape[0]}\n")
    return df

# ──────────────────────────────────────────────
# 2. MÉTRICAS DE PORTFOLIO
# ──────────────────────────────────────────────
def portfolio_metrics(weights, returns, cov_matrix, rf=RISK_FREE_RATE):
    w   = np.array(weights)
    ret = np.sum(returns.mean() * w) * 252
    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix * 252, w)))
    sharpe = (ret - rf) / vol if vol > 0 else 0
    return ret, vol, sharpe

# ──────────────────────────────────────────────
# 3. MONTE CARLO — FRONTEIRA EFICIENTE
# ──────────────────────────────────────────────
def monte_carlo_simulation(returns, cov_matrix, n=N_PORTFOLIOS):
    n_assets = returns.shape[1]
    results  = np.zeros((3, n))
    weights_all = np.zeros((n, n_assets))

    print(f"🎲 A simular {n:,} portfolios aleatórios...")
    for i in range(n):
        w = np.random.random(n_assets)
        w /= w.sum()
        ret, vol, sharpe = portfolio_metrics(w, returns, cov_matrix)
        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = sharpe
        weights_all[i] = w

    return results, weights_all

# ──────────────────────────────────────────────
# 4. OPTIMIZAÇÃO
# ──────────────────────────────────────────────
def optimize_portfolio(returns, cov_matrix, objective="sharpe"):
    n = returns.shape[1]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.01, 0.50) for _ in range(n))   # max 50% por activo
    w0 = np.array([1/n] * n)

    def neg_sharpe(w):
        r, v, s = portfolio_metrics(w, returns, cov_matrix)
        return -s

    def portfolio_vol(w):
        return portfolio_metrics(w, returns, cov_matrix)[1]

    def neg_return(w):
        return -portfolio_metrics(w, returns, cov_matrix)[0]

    obj_map = {
        "sharpe":  neg_sharpe,
        "min_risk": portfolio_vol,
        "max_return": neg_return,
    }

    result = minimize(obj_map[objective], w0,
                      method="SLSQP",
                      bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": 1000})
    return result.x

# ──────────────────────────────────────────────
# 5. VISUALIZAÇÕES
# ──────────────────────────────────────────────
def plot_efficient_frontier(results, weights_all, opt_portfolios, asset_names):
    fig, ax = plt.subplots(figsize=(13, 8))

    # Scatter Monte Carlo
    sc = ax.scatter(results[1]*100, results[0]*100,
                    c=results[2], cmap="RdYlGn",
                    alpha=0.4, s=8, zorder=1)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    # Portolios optimizados
    markers_cfg = [
        ("sharpe",      "⭐", "#FFD700", "Max Sharpe",     200, "*"),
        ("min_risk",    "🛡",  "#00BFFF", "Min Risk",       200, "D"),
        ("max_return",  "🚀", "#FF6B6B", "Max Return",     200, "^"),
    ]
    for key, emoji, color, label, size, marker in markers_cfg:
        if key not in opt_portfolios: continue
        w   = opt_portfolios[key]["weights"]
        ret = opt_portfolios[key]["return"] * 100
        vol = opt_portfolios[key]["volatility"] * 100
        ax.scatter(vol, ret, color=color, s=size, marker=marker,
                   zorder=5, label=label, edgecolors="black", linewidth=1.5)
        ax.annotate(f"  {label}\n  {ret:.1f}% ret | {vol:.1f}% vol",
                    (vol, ret), fontsize=8.5, color=color,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=color, alpha=0.85))

    ax.set_xlabel("Annualised Volatility (%)", fontsize=11)
    ax.set_ylabel("Annualised Return (%)",     fontsize=11)
    ax.set_title("Efficient Frontier — Portfolio Optimisation\n"
                 "BTC · ETH · S&P 500 · Gold · Silver · Apple · Palantir · US Treasuries",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_efficient_frontier.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  💾 1_efficient_frontier.png")


def plot_allocations(opt_portfolios, asset_names):
    objectives = list(opt_portfolios.keys())
    labels_map = {
        "sharpe":     "Max Sharpe",
        "min_risk":   "Min Risk",
        "max_return": "Max Return",
    }
    colors = [ASSETS[a]["color"] for a in asset_names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Optimal Portfolio Allocations", fontsize=14, fontweight="bold")

    for i, obj in enumerate(objectives):
        ax    = axes[i]
        w     = opt_portfolios[obj]["weights"]
        label = labels_map.get(obj, obj)

        # Filtrar pesos > 1%
        idx    = [j for j, x in enumerate(w) if x > 0.01]
        names  = [asset_names[j] for j in idx]
        vals   = [w[j] for j in idx]
        cols   = [colors[j] for j in idx]

        wedges, texts, autotexts = ax.pie(
            vals, labels=names, colors=cols,
            autopct="%1.1f%%", startangle=90,
            pctdistance=0.82,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2)
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_fontweight("bold")

        ret = opt_portfolios[obj]["return"] * 100
        vol = opt_portfolios[obj]["volatility"] * 100
        sh  = opt_portfolios[obj]["sharpe"]
        ax.set_title(f"{label}\nReturn: {ret:.1f}%  |  Vol: {vol:.1f}%  |  Sharpe: {sh:.2f}",
                     fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_allocations.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  💾 2_allocations.png")


def plot_weights_comparison(opt_portfolios, asset_names):
    labels_map = {"sharpe":"Max Sharpe","min_risk":"Min Risk","max_return":"Max Return"}
    objectives = list(opt_portfolios.keys())
    x = np.arange(len(asset_names))
    width = 0.25
    bar_colors = ["#FFD700", "#00BFFF", "#FF6B6B"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, obj in enumerate(objectives):
        w = opt_portfolios[obj]["weights"] * 100
        ax.bar(x + i*width, w, width, label=labels_map.get(obj,obj),
               color=bar_colors[i], alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels(asset_names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Allocation (%)")
    ax.set_title("Asset Allocation Comparison — 3 Objectives", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_weights_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  💾 3_weights_comparison.png")


def plot_cumulative_returns(prices, opt_portfolios, asset_names):
    returns = prices.pct_change().dropna()
    fig, ax = plt.subplots(figsize=(14, 7))

    colors_map = {"sharpe":"#FFD700","min_risk":"#00BFFF","max_return":"#FF6B6B"}
    labels_map = {"sharpe":"Max Sharpe","min_risk":"Min Risk","max_return":"Max Return"}

    for obj, cfg in opt_portfolios.items():
        w          = cfg["weights"]
        port_ret   = returns[asset_names].fillna(0).dot(w)
        cum_ret    = (1 + port_ret).cumprod() * INITIAL_CAPITAL
        ax.plot(cum_ret.index, cum_ret.values,
                label=labels_map[obj], color=colors_map[obj],
                linewidth=2.5)

    # Equal weight benchmark
    eq_w     = np.array([1/len(asset_names)] * len(asset_names))
    eq_ret   = returns[asset_names].fillna(0).dot(eq_w)
    eq_cum   = (1 + eq_ret).cumprod() * INITIAL_CAPITAL
    ax.plot(eq_cum.index, eq_cum.values,
            label="Equal Weight", color="#94A3B8",
            linewidth=1.5, linestyle="--")

    ax.axhline(INITIAL_CAPITAL, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title("Portfolio Performance — $10,000 Initial Capital (2 Years)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  💾 4_cumulative_returns.png")


def plot_correlation_heatmap(prices):
    returns = prices.pct_change().dropna()
    corr    = returns.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                annot_kws={"size": 9, "fontweight": "bold"})
    ax.set_title("Asset Correlation Matrix\n(Daily Returns — 2 Years)",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_correlation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  💾 5_correlation.png")


# ──────────────────────────────────────────────
# 6. RELATÓRIO FINAL
# ──────────────────────────────────────────────
def print_report(opt_portfolios, asset_names):
    labels = {"sharpe":"Max Sharpe","min_risk":"Min Risk","max_return":"Max Return"}
    print("\n" + "="*65)
    print("  PORTFOLIO OPTIMISATION — FINAL REPORT")
    print(f"  Assets: {', '.join(asset_names)}")
    print("="*65)

    for obj, res in opt_portfolios.items():
        print(f"\n  🎯 {labels[obj].upper()}")
        print(f"     Return     : {res['return']*100:+.2f}%")
        print(f"     Volatility : {res['volatility']*100:.2f}%")
        print(f"     Sharpe     : {res['sharpe']:.3f}")
        print(f"     Final Value: ${INITIAL_CAPITAL * (1 + res['return']):,.0f}")
        print(f"\n     Allocation:")
        for name, w in sorted(zip(asset_names, res["weights"]),
                               key=lambda x: x[1], reverse=True):
            if w > 0.005:
                bar = "█" * int(w * 40)
                print(f"       {name:<15} {w*100:5.1f}%  {bar}")

    print("\n" + "="*65)
    print("⚠️  Past performance does not guarantee future results.")
    print("   Not financial advice. For educational purposes only.\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("="*65)
    print("  📊 Portfolio Optimizer — Markowitz Modern Portfolio Theory")
    print("="*65)

    # 1. Dados
    prices = load_prices()
    asset_names = list(prices.columns)
    returns     = prices.pct_change().dropna()
    cov_matrix  = returns.cov()

    # 2. Monte Carlo
    mc_results, mc_weights = monte_carlo_simulation(returns, cov_matrix)
    print(f"  ✅ {N_PORTFOLIOS:,} portfolios simulados\n")

    # 3. Optimização
    print("🔧 A optimizar portfolios...\n")
    opt_portfolios = {}
    for obj in ["sharpe", "min_risk", "max_return"]:
        w   = optimize_portfolio(returns, cov_matrix, obj)
        ret, vol, sh = portfolio_metrics(w, returns, cov_matrix)
        opt_portfolios[obj] = {
            "weights":    w,
            "return":     ret,
            "volatility": vol,
            "sharpe":     sh,
        }
        label = {"sharpe":"Max Sharpe","min_risk":"Min Risk","max_return":"Max Return"}[obj]
        print(f"  ✅ {label}: Return={ret*100:+.1f}% | Vol={vol*100:.1f}% | Sharpe={sh:.2f}")

    # 4. Gráficos
    print("\n📈 A gerar gráficos...\n")
    plot_efficient_frontier(mc_results, mc_weights, opt_portfolios, asset_names)
    plot_allocations(opt_portfolios, asset_names)
    plot_weights_comparison(opt_portfolios, asset_names)
    plot_cumulative_returns(prices, opt_portfolios, asset_names)
    plot_correlation_heatmap(prices)

    # 5. Relatório
    print_report(opt_portfolios, asset_names)

    print("✅ Portfolio optimisation complete!\n")
