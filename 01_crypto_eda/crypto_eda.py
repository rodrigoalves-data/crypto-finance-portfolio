"""
=====================================================================
 EDA de Ativos Cripto de Topo — Portfólio de Análise de Dados
=====================================================================
Objetivo  : Analisar correlação de preços e volatilidade entre
            BTC, ETH, SOL, BNB e XRP no último ano.
Fonte     : CoinGecko API (pública, sem chave)
Autor     : <O teu nome>
Data      : 2026
=====================================================================
Dependências:
    pip install requests pandas matplotlib seaborn
=====================================================================
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────
COINS = {
    "bitcoin":   "BTC",
    "ethereum":  "ETH",
    "solana":    "SOL",
    "binancecoin": "BNB",
    "ripple":    "XRP",
}

CURRENCY   = "usd"
DAYS       = 365          # último ano
OUTPUT_DIR = "."          # pasta de saída dos gráficos

# Paleta visual consistente
PALETTE = {
    "BTC": "#F7931A",
    "ETH": "#627EEA",
    "SOL": "#9945FF",
    "BNB": "#F0B90B",
    "XRP": "#00AAE4",
}


# ──────────────────────────────────────────────
# 1. COLETA DE DADOS — CoinGecko API
# ──────────────────────────────────────────────
def fetch_price_history(coin_id: str, vs_currency: str = "usd", days: int = 365) -> pd.Series:
    """Busca o histórico de preços diários de uma moeda via CoinGecko."""
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        f"?vs_currency={vs_currency}&days={days}&interval=daily"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    prices["date"] = pd.to_datetime(prices["timestamp"], unit="ms").dt.normalize()
    prices = prices.drop_duplicates("date").set_index("date")["price"]
    prices.name = COINS[coin_id]
    return prices


def load_all_prices() -> pd.DataFrame:
    """Coleta e consolida preços de todas as moedas."""
    all_series = []
    print("📡 A buscar dados da CoinGecko...\n")
    for coin_id, ticker in COINS.items():
        try:
            s = fetch_price_history(coin_id, CURRENCY, DAYS)
            all_series.append(s)
            print(f"  ✅ {ticker} — {len(s)} dias carregados")
        except Exception as e:
            print(f"  ❌ {ticker} — Erro: {e}")
        time.sleep(1)   # respeita o rate-limit da API pública

    df = pd.concat(all_series, axis=1).sort_index()
    return df


# ──────────────────────────────────────────────
# 2. LIMPEZA & VALIDAÇÃO DOS DADOS
# ──────────────────────────────────────────────
def clean_data(df: pd.DataFrame, liquidity_threshold: float = 0.20) -> pd.DataFrame:
    """
    Remove:
      - Datas com mais de X% de valores em falta (sem liquidez / dados corrompidos).
      - Colunas (moedas) com dados insuficientes.
    Preenche lacunas pontuais com forward-fill + backward-fill.
    """
    print("\n🧹 A limpar dados...")

    # Verifica % de NaN por coluna
    missing_pct = df.isnull().mean()
    to_drop = missing_pct[missing_pct > liquidity_threshold].index.tolist()
    if to_drop:
        print(f"  ⚠️  Removidas moedas com >{ liquidity_threshold*100:.0f}% de dados em falta: {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        print("  ✅ Todas as moedas passaram no filtro de liquidez.")

    # Preenche lacunas menores
    df = df.ffill().bfill()

    # Remove datas sem qualquer dado
    df = df.dropna(how="all")

    print(f"  📅 Período final: {df.index.min().date()} → {df.index.max().date()}")
    print(f"  📊 Forma do dataset: {df.shape[0]} dias × {df.shape[1]} activos\n")
    return df


# ──────────────────────────────────────────────
# 3. MÉTRICAS
# ──────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calcula retornos diários, retornos normalizados e estatísticas de volatilidade."""

    # Retornos diários (variação percentual)
    returns = df.pct_change().dropna()

    # Preços normalizados (base 100 no início do período)
    normalized = (df / df.iloc[0]) * 100

    # Estatísticas de volatilidade
    stats = pd.DataFrame({
        "Volatilidade Diária (σ)":   returns.std(),
        "Volatilidade Anualizada":   returns.std() * np.sqrt(365),
        "Retorno Total (%)":         ((df.iloc[-1] / df.iloc[0]) - 1) * 100,
        "Retorno Médio Diário (%)":  returns.mean() * 100,
        "Sharpe Simplificado":       (returns.mean() / returns.std()) * np.sqrt(365),
        "Max Drawdown (%)":          ((df / df.cummax()) - 1).min() * 100,
    }).round(4)

    return returns, normalized, stats


# ──────────────────────────────────────────────
# 4. VISUALIZAÇÕES
# ──────────────────────────────────────────────
def plot_normalized_prices(normalized: pd.DataFrame):
    """Evolução do preço normalizado (base 100)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for col in normalized.columns:
        ax.plot(normalized.index, normalized[col],
                label=col, color=PALETTE.get(col, None), linewidth=2)

    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Base (dia 0)")
    ax.set_title("Evolução do Preço Normalizado (Base 100)", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("Índice de Preço (Base = 100)")
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_precos_normalizados.png", dpi=150)
    plt.show()
    print("  💾 Gráfico guardado: 1_precos_normalizados.png")


def plot_correlation_matrix(returns: pd.DataFrame):
    """Matriz de correlação de Pearson entre os retornos diários."""
    corr = returns.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # mascara triângulo superior

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1, vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 12, "fontweight": "bold"},
    )
    ax.set_title("Matriz de Correlação de Pearson\n(Retornos Diários — Último Ano)",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_matriz_correlacao.png", dpi=150)
    plt.show()
    print("  💾 Gráfico guardado: 2_matriz_correlacao.png")
    return corr


def plot_volatility_comparison(returns: pd.DataFrame):
    """Gráfico de barras comparando a volatilidade anualizada."""
    vol_annual = (returns.std() * np.sqrt(365) * 100).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(vol_annual.index, vol_annual.values,
                  color=[PALETTE.get(c, "#999") for c in vol_annual.index],
                  edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, vol_annual.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_title("Volatilidade Anualizada por Activo (%)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Volatilidade Anualizada (%)")
    ax.set_ylim(0, vol_annual.max() * 1.2)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_volatilidade.png", dpi=150)
    plt.show()
    print("  💾 Gráfico guardado: 3_volatilidade.png")


def plot_rolling_volatility(returns: pd.DataFrame, window: int = 30):
    """Volatilidade móvel de 30 dias para cada activo."""
    rolling_vol = returns.rolling(window).std() * np.sqrt(365) * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    for col in rolling_vol.columns:
        ax.plot(rolling_vol.index, rolling_vol[col],
                label=col, color=PALETTE.get(col, None), linewidth=1.8, alpha=0.85)

    ax.set_title(f"Volatilidade Anualizada Móvel ({window} dias)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Volatilidade Anualizada (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_volatilidade_movel.png", dpi=150)
    plt.show()
    print("  💾 Gráfico guardado: 4_volatilidade_movel.png")


def plot_return_distribution(returns: pd.DataFrame):
    """Distribuição dos retornos diários (histograma + KDE)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(returns.columns):
        ax = axes[i]
        sns.histplot(returns[col] * 100, bins=50, kde=True,
                     color=PALETTE.get(col, "#555"), ax=ax, alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{col} — Distribuição dos Retornos Diários",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Retorno Diário (%)")
        ax.set_ylabel("Frequência")
        mean = (returns[col] * 100).mean()
        std  = (returns[col] * 100).std()
        ax.legend([f"μ={mean:.2f}%  σ={std:.2f}%"], fontsize=9)

    # esconde o subplot extra se houver
    for j in range(len(returns.columns), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribuição dos Retornos Diários por Activo",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_distribuicao_retornos.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  💾 Gráfico guardado: 5_distribuicao_retornos.png")


# ──────────────────────────────────────────────
# 5. RELATÓRIO SUMÁRIO NO TERMINAL
# ──────────────────────────────────────────────
def print_summary(stats: pd.DataFrame, corr: pd.DataFrame):
    print("\n" + "="*60)
    print("  SUMÁRIO — EDA CRIPTO (ÚLTIMO ANO)")
    print("="*60)
    print("\n📊 ESTATÍSTICAS POR ACTIVO:\n")
    print(stats.to_string())

    print("\n\n🔗 CORRELAÇÕES DE PEARSON (top pares):\n")
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .rename(columns={"level_0": "Activo A", "level_1": "Activo B", 0: "Correlação"})
        .sort_values("Correlação", ascending=False)
    )
    print(pairs.to_string(index=False))

    # Insights automáticos
    most_volatile = stats["Volatilidade Anualizada"].idxmax()
    least_volatile = stats["Volatilidade Anualizada"].idxmin()
    best_return    = stats["Retorno Total (%)"].idxmax()
    highest_corr   = pairs.iloc[0]

    print("\n\n💡 INSIGHTS AUTOMÁTICOS:")
    print(f"  • Activo mais volátil   : {most_volatile}")
    print(f"  • Activo menos volátil  : {least_volatile}")
    print(f"  • Melhor retorno no ano : {best_return} ({stats.loc[best_return,'Retorno Total (%)']:.1f}%)")
    print(f"  • Par mais correlacionado: {highest_corr['Activo A']} ↔ {highest_corr['Activo B']} "
          f"(r = {highest_corr['Correlação']:.2f})")
    print("\n" + "="*60)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  🚀 EDA de Ativos Cripto de Topo")
    print("="*60)

    # 1. Carregar dados
    prices = load_all_prices()

    # 2. Limpar
    prices = clean_data(prices)

    # 3. Métricas
    returns, normalized, stats = compute_metrics(prices)

    # 4. Visualizações
    print("\n📈 A gerar gráficos...\n")
    plot_normalized_prices(normalized)
    corr = plot_correlation_matrix(returns)
    plot_volatility_comparison(returns)
    plot_rolling_volatility(returns)
    plot_return_distribution(returns)

    # 5. Sumário
    print_summary(stats, corr)

    print("\n✅ Análise concluída! Todos os gráficos foram guardados na pasta actual.\n")
