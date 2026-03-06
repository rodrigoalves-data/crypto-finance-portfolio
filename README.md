# 📊 Crypto & Quantitative Finance Portfolio

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Status](https://img.shields.io/badge/Status-Active-00FF88?style=flat)

**Author:** Rodrigo Ferreira Alves
**Focus:** Data Science applied to Quantitative Finance, Algorithmic Trading & Crypto Markets
**Stack:** Python · Pandas · NumPy · TensorFlow · Scipy · REST APIs · AI-Augmented Workflows

---

## 🗂️ Projects

### 01 — Crypto EDA (Exploratory Data Analysis)
> Comprehensive analysis of BTC, ETH, SOL, BNB and XRP over 365 days.

- Price normalisation, daily returns and annualised volatility
- Pearson correlation matrix across all asset pairs
- Sharpe Ratio and Maximum Drawdown per asset
- Automated insight generation from statistical results

**Key finding:** All assets showed high correlation (0.68–0.87), limiting diversification benefits within crypto.
**APIs:** CoinGecko

---

### 02 — LSTM Price Prediction
> Deep learning model for multi-asset price forecasting using Long Short-Term Memory networks.

- Multi-source data: CoinGecko (crypto) + Yahoo Finance (stocks/commodities)
- 2-layer LSTM architecture (128 → 64 neurons) with Dropout regularisation
- 60-day lookback window with 80/20 train/test split
- Early stopping to prevent overfitting
- 30-day recursive forecasting with confidence bands
- Evaluation: MAE, RMSE, MAPE

**Assets:** BTC, ETH, RIOT, Crude Oil

---

### 03 — Live Market Dashboard
> Real-time web dashboard with AI-powered market signal generation.

- Live price feeds: BTC, ETH, Gold (XAU/USD), S&P 500
- News sentiment scoring using NLP keyword analysis
- Fear & Greed Index integration
- AI composite signal: BULLISH / NEUTRAL / BEARISH
- Auto-refresh every 5 minutes
- Single-file HTML — deployable via Netlify Drop in 30 seconds

**APIs:** CoinGecko · Alternative.me · CryptoPanic

---

### 04 — Liquidity Sweep Backtesting
> Algorithmic backtesting engine based on Smart Money Concepts (SMC).

- Real OHLCV data via KuCoin API (4H and 1H timeframes)
- Automatic Swing High / Swing Low detection
- Bullish & Bearish Liquidity Sweep identification
- Trade simulation with 1:3 Risk:Reward ratio and 2% risk per trade
- Metrics: Win Rate, Sharpe Ratio, Max Drawdown, Profit Factor, Expectancy

**Assets:** BTC/USDT · ETH/USDT
**Timeframes:** 4H · 1H · 3 years of data

---

### 05 — Portfolio Optimizer
> Modern Portfolio Theory implementation with Monte Carlo simulation and constrained optimisation.

- 10,000 random portfolio simulations (Monte Carlo)
- Three optimisation objectives via Scipy:
  - **Max Sharpe Ratio** — best risk-adjusted return
  - **Min Volatility** — most defensive allocation
  - **Max Return** — most aggressive allocation
- Efficient Frontier visualisation
- Cumulative performance vs Equal Weight benchmark
- Full correlation matrix across all assets

**Assets:** BTC · ETH · S&P 500 · Gold · Silver · Apple · Palantir · US Treasuries

---

### 06 — On-Chain Analysis
> Blockchain data analysis combining network activity, whale behaviour and institutional ETF flows.

- BTC Active Addresses trend and price correlation
- Whale Activity Ratio (average transaction value proxy)
- Spot ETF cumulative flows — BTC vs ETH
- Composite On-Chain Signal Score (0–100):
  - 30% Active Addresses · 30% Whale Activity · 40% ETF Flows
- Signal zones overlaid on price chart (Bullish > 65 | Bearish < 35)

**APIs:** Blockchain.com · Etherscan · CoinGecko

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Data Collection | CoinGecko API, Yahoo Finance, Blockchain.com, KuCoin API, Etherscan |
| Data Processing | Pandas, NumPy |
| Machine Learning | TensorFlow / Keras (LSTM) |
| Optimisation | Scipy (SLSQP) |
| Visualisation | Matplotlib, Seaborn |
| Web | HTML, CSS, JavaScript, Chart.js |
| Workflow | AI-Augmented (Python + LLM assistance) |

---

## 📈 Skills Demonstrated

- Time series analysis and forecasting
- Deep learning for financial data (LSTM)
- Quantitative backtesting with realistic trade simulation
- Portfolio optimisation (Markowitz, Monte Carlo)
- On-chain / blockchain data analysis
- REST API integration (multiple sources)
- Signal generation and composite scoring
- Data visualisation and dashboard development

---

## ⚠️ Disclaimer

All projects in this repository are for **educational and portfolio purposes only**.
Nothing here constitutes financial advice.
Past performance does not guarantee future results.

---

*Rodrigo Ferreira Alves — 2026*
