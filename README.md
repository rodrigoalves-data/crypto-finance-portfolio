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

### 07 — Ensemble Price Prediction Model (LSTM + XGBoost + Random Forest)
> Combining LSTM, XGBoost and Random Forest for multi-asset price forecasting.

- 20 technical features per asset: RSI, MACD, Bollinger Bands, momentum, volume ratio
- Three independent models trained on identical data, each learning different patterns
- Weighted ensemble — models with lower MAE receive higher weight automatically
- Directional Accuracy as primary metric — measures if the model predicts up/down correctly

**Results (Test Set — 60 days):**

| Asset | Best Model | MAPE | Directional Accuracy |
|---|---|---|---|
| BTC | LSTM | 4.3% | **59.0%** ✅ |
| ETH | RF | 3.7% | 48.7% |
| AAPL | XGBoost | 1.6% | 56.4% ✅ |
| EUR/USD | LSTM / RF | 0.6% | 53.8% |

> Directional Accuracy above 55% is considered actionable for trading signals.
> BTC (59%) and AAPL (56.4%) crossed this threshold.

**Tech stack:** Python · TensorFlow/Keras · XGBoost · Scikit-learn · Yahoo Finance API

---

### 08 — RTH Gap Fill Statistics (NQ & ES Futures)
> Statistical analysis of Regular Trading Hours gaps on Nasdaq 100 and S&P 500 futures over 5 years.

- Identifies all bullish and bearish RTH gaps (open vs previous close)
- Calculates fill rates at 25%, 50%, 75% and 100% on the same day
- Measures 50% fill rate in the first hour of trading (09:30–10:30 ET)
- Tracks average days to full gap close across 5 years
- Comparative dashboard: NQ vs ES side by side

**Key Results (NQ):**

| Gap Type | Fill 25% | Fill 50% | Fill 100% | First Hour 50% |
|---|---|---|---|---|
| Bullish | 96.1% | 92.9% | 86.9% | 70.8% |
| Bearish | 97.2% | 94.5% | 89.4% | 77.9% |

> Fill rate above 70% = strong statistical edge for intraday trading.

**Tech stack:** Python · Pandas · Matplotlib · Yahoo Finance API

---

### 09 — Order Flow Analysis (NQ, ES, BTC, ETH)
> Institutional-grade order flow analysis combining VWAP, Volume Profile and Cumulative Delta.

- **VWAP** with daily reset and ±1σ / ±2σ standard deviation bands
- **Volume Profile** — POC, VAH, VAL with 70% Value Area calculation
- **Rolling Volume Profile** — 20-day window showing how key levels evolve
- **Cumulative Delta** — buy vs sell volume pressure estimation
- **Divergence detection** — price/delta divergence as reversal signal
- Combined dashboard: Price + VWAP + Volume Profile + Delta in one view

**Assets:** NQ · ES · BTC · ETH
**Tech stack:** Python · Pandas · NumPy · Matplotlib · Yahoo Finance API

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
