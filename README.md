# Binance Futures Trading Bot

A professional automated trading bot for Binance Futures with **Hybrid AI** (Expert System + Machine Learning), automatic training, multi-timeframe analysis, and advanced risk management.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)]()

---

## 🚀 Key Features

- **🤖 Hybrid AI Strategy**: Combines Experta rule-based system (explainable) with ML pattern recognition (powerful)
- **⚡ Fully Automatic Training**: Bot automatically downloads data and trains ML models on first run - zero configuration!
- **💾 Quick Restart**: Saves your configuration and offers to reuse it on restart - 10x faster restarts!
- **📦 Smart Data Management**: Uses cached data when available, downloads from Binance when needed
- **🎯 Multi-Symbol/Timeframe**: Separate trained models for each symbol-timeframe combination
- **📊 45+ Technical Indicators**: RSI, MACD, EMA, ADX, Supertrend, Ichimoku, and more (pandas_ta)
- **🧠 Expert System**: 40+ intelligent trading rules with priority-based execution
- **📈 Backtesting Engine**: Test strategies on historical data with detailed performance metrics
- **🛡️ Risk Management**: Position sizing, stop-loss, take-profit, trailing stops, daily loss limits
- **💾 Database Logging**: SQLite for tracking all trades and bot activity
- **🔄 Real-time Trading**: REST API polling for stable, reliable data

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#️-configuration)
- [How It Works](#-how-it-works)
- [Hybrid AI Strategy](#-hybrid-ai-strategy)
- [Automatic Training](#-automatic-training)
- [Backtesting](#-backtesting)
- [Risk Management](#️-risk-management)
- [Project Structure](#-project-structure)
- [Advanced Usage](#-advanced-usage)
- [FAQ](#-faq)
- [License](#-license)

---

## 🎯 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/binance-bot.git
cd binance-bot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp env.example .env
# Edit .env with your Binance API keys

# 5. Run the bot
python main.py
```

**That's it!** 🎉 On first run, the bot will automatically:
- Check for cached historical data
- Download data from Binance if needed
- Train an ML model (30-120 seconds)
- Start trading with full Hybrid AI

---

## 💻 Installation

### Requirements

- Python 3.12+
- Binance Futures Account (Testnet or Live)
- API Key with Futures permissions

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `python-binance` - Binance API wrapper
- `pandas` & `numpy` - Data manipulation
- `pandas-ta` - Technical indicators
- `experta` - Expert system (rule engine)
- `scikit-learn` - Machine learning
- `joblib` - Model persistence
- `pyarrow` - Efficient data storage

### Step 2: Get Binance API Keys

**For Testing (Recommended):**
1. Visit [Binance Futures Testnet](https://testnet.binancefuture.com/)
2. Create an account (use any email, no verification needed)
3. Generate API Key with Futures permissions

**For Live Trading:**
1. Visit [Binance](https://www.binance.com/)
2. Create account and complete KYC
3. Enable Futures trading
4. Generate API Key with **"Enable Futures"** permission
5. Whitelist IP address (optional but recommended)

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```bash
# API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
TESTNET=true  # Set to false for live trading

# Trading Parameters
TRADING_PAIR=BTCUSDT
TIMEFRAME=5m
LEVERAGE=3
MARGIN_TYPE=ISOLATED

# Risk Management
POSITION_SIZE_PCT=90      # Use 90% of available balance
STOP_LOSS_PCT=2.0         # Stop loss at -2%
TAKE_PROFIT_PCT=5.0       # Take profit at +5%
TRAILING_STOP_PCT=1.5     # Trailing stop distance
MAX_DAILY_LOSS_PCT=5.0    # Max daily loss before stopping

# Signal Settings
MIN_CONFIDENCE=55         # Minimum signal confidence (0-100)

# Multi-Timeframe Analysis
USE_MULTI_TIMEFRAME=true  # Enable higher timeframe analysis
```

### Configuration Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `TRADING_PAIR` | Symbol to trade | BTCUSDT | Any Binance Futures pair |
| `TIMEFRAME` | Candle timeframe | 5m | 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 1d |
| `LEVERAGE` | Leverage multiplier | 3 | 1-125 |
| `POSITION_SIZE_PCT` | % of balance per trade | 90 | 1-100 |
| `STOP_LOSS_PCT` | Stop loss percentage | 2.0 | 0.5-10 |
| `TAKE_PROFIT_PCT` | Take profit percentage | 5.0 | 1-50 |
| `TRAILING_STOP_PCT` | Trailing stop distance | 1.5 | 0.5-5 |
| `MIN_CONFIDENCE` | Signal threshold | 55 | 0-100 |

---

## 🧠 How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID AI STRATEGY                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────┐     ┌──────────────────────┐    │
│  │  EXPERTA RULES       │     │  ML ENHANCEMENT      │    │
│  │  (Explainable AI)    │     │  (Pattern Learning)  │    │
│  │                      │     │                      │    │
│  │  • 40+ trading rules │     │  • Gradient Boosting │    │
│  │  • Priority-based    │     │  • 27 features       │    │
│  │  • Market regimes    │     │  • Symbol-specific   │    │
│  │  • 70% weight        │     │  • 30% weight        │    │
│  └──────────────────────┘     └──────────────────────┘    │
│           │                              │                 │
│           └──────────────┬───────────────┘                 │
│                          ▼                                 │
│              ┌────────────────────┐                        │
│              │   FINAL SIGNAL     │                        │
│              │  Confidence: 72.5% │                        │
│              └────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
         ↓
    Market Data (Binance API)
         ↓
    45+ Technical Indicators (pandas_ta)
         ↓
    Risk Management & Position Sizing
         ↓
    Trade Execution
```

### Trading Flow

1. **Data Collection**: Fetch latest candles from Binance (REST API)
2. **Indicator Calculation**: Compute 45+ technical indicators
3. **Market Regime Detection**: Identify current market conditions (TRENDING/RANGING/VOLATILE)
4. **Signal Generation**: 
   - Experta rules fire based on indicators (70% weight)
   - ML model predicts signal success (30% weight)
   - Combine for final confidence score
5. **Risk Check**: Verify signal meets confidence threshold and risk parameters
6. **Position Sizing**: Calculate position size based on available balance and leverage
7. **Order Execution**: Place limit order on Binance (with smart retry system)
8. **Position Management**: Monitor with trailing stops, take-profit, and stop-loss

---

## 🤖 Hybrid AI Strategy

### Dual-Layer AI System

Our bot uses a unique **Hybrid AI** approach that combines the best of both worlds:

#### Layer 1: Experta Expert System (70% weight)

**Explainable, transparent, rule-based decision making.**

- **40+ Trading Rules** organized by category
- **Priority-based execution** (salience system)
- **Market regime detection** (TRENDING, RANGING, VOLATILE)
- **Transparent reasoning** - you always know why a signal was generated

**Rule Categories:**

1. **Market Regime Detection** - Identify market conditions
2. **Trend Following** - Perfect uptrend/downtrend patterns
3. **Mean Reversion** - Overbought/oversold in ranging markets
4. **Breakout Detection** - Support/resistance breakouts
5. **Volume Confirmation** - CMF + OBV validation
6. **Risk Filters** - Block trades in extreme conditions

**Example Rule:**
```python
@Rule(
    MarketCondition(regime='STRONG_TREND'),
    Indicator(name='adx', value=P(lambda x: x > 25)),
    Indicator(name='ema_9_above_21', value=True),
    salience=100  # High priority
)
def perfect_uptrend(self):
    self.add_signal('BUY', 40, 'Perfect uptrend setup')
```

#### Layer 2: ML Enhancement (30% weight)

**Pattern recognition and confidence adjustment.**

- **Gradient Boosting Classifier** (scikit-learn)
- **27 Features**: RSI, ADX, MACD, CMF, MFI, trend score, etc.
- **Learns from 500-2000+ real market candles**
- **Symbol-specific models** (BTCUSDT ≠ ETHUSDT)
- **Confidence adjustment**: Boosts or reduces signal strength

**Training:**
- **Data**: 500-2000+ candles (cached or from Binance)
- **Samples**: 100-400+ training examples
- **Accuracy**: 70-85% on test set
- **Time**: 30-120 seconds to train

#### Confidence Blending

```python
final_confidence = (rule_confidence × 0.7) + (ml_probability × 0.3)

Example:
  Rule confidence: 65%
  ML probability:  85%
  Final: (65 × 0.7) + (85 × 0.3) = 45.5 + 25.5 = 71%
```

This approach provides:
- ✅ **Explainability** (from rules)
- ✅ **Pattern recognition** (from ML)
- ✅ **Bias correction** (ML learns from outcomes)
- ✅ **Best of both worlds**

---

## ⚡ Automatic Training

### Zero Manual Configuration

The bot includes **fully automatic ML model training**:

- **First run**: Bot detects missing model and automatically trains
- **No user interaction**: Downloads data and trains without asking
- **Smart data source**: Uses cached data if available, downloads if needed
- **Fast training**: 30-120 seconds using real Binance data
- **Automatic loading**: Next run loads model instantly

### How Auto-Training Works

```
1. Check for Cached Data
   ↓ Look in data/cached/{SYMBOL}_{TIMEFRAME}.parquet
   
2a. If cached data exists (≥500 candles)
   ↓ Use cached data (instant!)
   
2b. If no cached data
   ↓ Download from Binance (500-2000+ candles)
   ↓ Save to cache for future use
   
3. Calculate Indicators
   ↓ 45+ technical indicators (~5-10 sec)
   
4. Generate Training Samples
   ↓ 100-400+ signal examples with price-based labels
   
5. Train ML Model
   ↓ Gradient Boosting Classifier (~10-30 sec)
   
6. Save Model
   ↓ models/ml_{SYMBOL}_{TIMEFRAME}.pkl

Total: 30-120 seconds
```

### Multi-Symbol Support

Each symbol-timeframe gets its own model:

```
models/
  ml_BTCUSDT_5m.pkl     # Bitcoin 5-minute
  ml_ETHUSDT_15m.pkl    # Ethereum 15-minute
  ml_BNBUSDT_1h.pkl     # BNB 1-hour
```

**Why?** Each market has unique patterns. BTCUSDT 5m ≠ ETHUSDT 15m.

### Data Caching

Downloaded data is stored in Parquet format for fast reuse:

```
data/cached/
  BTCUSDT_5m.parquet    # 10,000+ candles, ~2 MB
  ETHUSDT_15m.parquet
  BNBUSDT_1h.parquet
```

Benefits:
- ⚡ **10x faster** than re-downloading
- 💾 **10x smaller** than CSV
- 🔄 **Reusable** for multiple training sessions

---

## 📊 Backtesting

Test your strategy on historical data before going live!

### Running a Backtest

```bash
python tools/backtest.py

# Prompts:
# Trading symbol [BTCUSDT]: BTCUSDT
# Timeframe [5m]: 5m
# Days of history [7]: 7
# Strategy: 2 (Hybrid)
```

### Backtest Output

```
📊 Performance:
   Initial Capital:  $10,000.00
   Final Capital:    $12,345.00
   Total P&L:        $+2,345.00 (+23.45%)

📈 Trade Statistics:
   Total Trades:     45
   Winning Trades:   28 (62.2%)
   Losing Trades:    17
   Average Win:      $154.32
   Average Loss:     $-68.21
   Profit Factor:    2.26

⚠️  Risk Metrics:
   Max Drawdown:     -8.34%
   Sharpe Ratio:     1.85

📝 Last 5 Trades:
   LONG  | Entry: $43,250 | Exit: $43,820 | P&L: +$128 (+1.3%)
   SHORT | Entry: $43,900 | Exit: $43,450 | P&L: +$95 (+1.0%)
   ...
```

### Features

- ✅ Realistic trade simulation
- ✅ Transaction fees (0.04%)
- ✅ Stop loss & take profit
- ✅ Comprehensive metrics (win rate, profit factor, Sharpe ratio)
- ✅ CSV export for analysis
- ✅ Compare Rules-only vs Hybrid

---

## 🛡️ Risk Management

### Built-in Risk Controls

1. **Position Sizing**
   - Uses % of available balance (default: 90%)
   - Accounts for leverage
   - Never over-leverages

2. **Stop Loss**
   - Hard stop at configured % (default: -2%)
   - Automatic order placement

3. **Take Profit**
   - Target profit % (default: +5%)
   - Locks in gains

4. **Trailing Stop**
   - Follows price up (for longs) or down (for shorts)
   - Distance: configurable % (default: 1.5%)
   - Maximizes profit on strong moves

5. **Daily Loss Limit**
   - Bot stops trading if daily loss exceeds limit (default: -5%)
   - Prevents catastrophic losses

6. **Minimum Confidence**
   - Only trades signals above threshold (default: 55%)
   - Quality over quantity

### Risk Configuration

Adjust in `.env`:

```bash
POSITION_SIZE_PCT=90      # Conservative: 50-70, Aggressive: 90-100
STOP_LOSS_PCT=2.0         # Tight: 1-2, Loose: 3-5
TAKE_PROFIT_PCT=5.0       # Conservative: 3-5, Aggressive: 7-10
TRAILING_STOP_PCT=1.5     # Tight: 0.5-1, Loose: 2-3
MAX_DAILY_LOSS_PCT=5.0    # Strict: 2-3, Relaxed: 5-10
```

---

## 📁 Project Structure

```
binance-bot/
├── Core Trading System
│   ├── main.py                 # Bot main loop with automatic training
│   ├── strategy.py             # Experta expert system (40+ rules)
│   ├── ml_model.py             # ML enhancement + auto-training logic
│   ├── indicators.py           # 45+ technical indicators
│   ├── binance_futures.py      # Binance API wrapper
│   ├── config.py               # Configuration management
│   ├── database.py             # SQLite database for trades/logs
│   └── risk_manager.py         # Position sizing & risk controls
│
├── Data Infrastructure
│   ├── data/
│   │   ├── binance_downloader.py  # Download historical data
│   │   ├── data_manager.py        # Load and manage cached data
│   │   └── cached/                # Parquet files (auto-created)
│   │       ├── BTCUSDT_5m.parquet
│   │       └── ...
│
├── ML Models
│   ├── models/                    # Trained ML models (auto-created)
│   │   ├── ml_BTCUSDT_5m.pkl
│   │   └── ...
│
├── Tools
│   └── tools/
│       └── backtest.py            # Backtesting engine
│
├── Configuration
│   ├── .env                       # Your configuration (create from .env.example)
│   ├── env.example                # Configuration template
│   ├── requirements.txt           # Python dependencies
│   └── bot.db                     # SQLite database (auto-created)
│
└── Documentation
    ├── README.md                  # This file
    └── LICENSE                    # MIT License
```

**Total:** ~3,500 lines of clean, production-ready Python code

---

## 🔧 Advanced Usage

### Configuration Cache (Quick Restart)

The bot automatically saves your configuration settings and offers to reuse them on restart:

**First Run:**
- Enter all your settings (trading pair, timeframe, leverage, etc.)
- Bot saves configuration to `config_cache.json`

**Second Run (and later):**
```
✓ Found saved configuration from previous session
  Last used: 2025-10-09T14:32:15

Previous Settings:
  • Trading Pair: BTCUSDT
  • Timeframe: 5m
  • Leverage: 3x
  ...

Use these settings? [Y/n]:
```

**Benefits:**
- ⚡ **10x faster restarts** - Just press Y
- ✅ **No typing errors** - Reuse proven settings
- 🔄 **Flexibility** - Press 'n' to reconfigure anytime
- 🔐 **Secure** - API keys never cached (always from .env)

To start fresh, delete the cache:
```bash
rm config_cache.json
```

### Manual Data Download (Optional)

While the bot downloads data automatically, you can pre-download for faster training:

```python
# Quick script to download data
from data.binance_downloader import BinanceDataDownloader

downloader = BinanceDataDownloader()

# Download single symbol
downloader.download_symbol('BTCUSDT', '5m')

# Download multiple symbols
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
intervals = ['5m', '15m', '1h']
downloader.download_multiple(symbols, intervals)
```

### Custom ML Features

Add your own features to `ml_model.py`:

```python
# In _extract_features() method
features['my_custom_indicator'] = row.get('my_indicator', 0)
```

### Custom Trading Rules

Add new rules to `strategy.py`:

```python
@Rule(
    Indicator(name='rsi', value=P(lambda x: x < 20)),
    Indicator(name='volume_spike', value=True)
)
def my_custom_rule(self):
    self.add_signal('BUY', 35, 'Custom pattern detected')
```

### Database Queries

Access trade history:

```python
from database import Database

db = Database('bot.db')

# Get all trades
trades = db.conn.execute("SELECT * FROM trades").fetchall()

# Get today's P&L
pnl = db.get_daily_pnl()
```

---

## ❓ FAQ

### General

**Q: Is this profitable?**  
A: Trading results vary based on market conditions, settings, and risk management. Always backtest thoroughly and start with small amounts on testnet.

**Q: Do I need ML experience?**  
A: No! The bot handles everything automatically. Just configure .env and run.

**Q: Can I use this on live trading?**  
A: Yes, but start with testnet first. Set `TESTNET=false` in `.env` when ready.

**Q: How much capital do I need?**  
A: Minimum $100 on testnet for testing. $1000+ recommended for live trading.

### Technical

**Q: How accurate is the ML model?**  
A: Typical test accuracy: 70-85%. Combined with rules, provides robust signals.

**Q: How often should I retrain models?**  
A: Models automatically train on first run. Manually retrain weekly or after major market changes by deleting the model file.

**Q: Can I add my own rules?**  
A: Yes! Edit `strategy.py` and add new `@Rule` decorators.

**Q: What about funding rates?**  
A: Bot tracks them. On average, ~0.01% per 8 hours (~10% annually). Factor into strategy.

**Q: Does it work with all symbols?**  
A: Yes, any Binance Futures pair. Each symbol-timeframe gets its own trained model.

### Trading

**Q: What's the best timeframe?**  
A: 5m-15m for active trading, 1h-4h for swing trading. Backtest to find your preference.

**Q: Can I trade multiple symbols?**  
A: Currently one symbol per bot instance. Run multiple instances for multiple symbols.

**Q: How do I change symbols?**  
A: Update `TRADING_PAIR` and `TIMEFRAME` in `.env`. Bot will auto-train a new model on next run.

**Q: Do I need to re-enter configuration every time I restart?**  
A: No! The bot saves your settings to `config_cache.json` and asks if you want to reuse them. Just press Y for instant restart.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**IMPORTANT**: Trading cryptocurrencies involves significant risk. This bot is for educational and research purposes. Use at your own risk.

- Past performance is not indicative of future results
- Never invest more than you can afford to lose
- Thoroughly test on testnet before live trading
- The developers are not responsible for any financial losses
- Cryptocurrency trading may be regulated in your jurisdiction

---

## 🌟 Acknowledgments

- **Binance** for providing the Futures API
- **pandas_ta** for technical indicators
- **Experta** for the expert system framework
- **scikit-learn** for machine learning tools

---

## 📞 Support

Have questions or found a bug?

- **Issues**: [GitHub Issues](https://github.com/yourusername/binance-bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/binance-bot/discussions)

---

## 🎯 Quick Reference

### Common Commands

```bash
# Run bot
python main.py

# Backtest strategy
python tools/backtest.py
```

### Key Metrics

- **ML Accuracy**: 70-85% (varies by market)
- **Win Rate Target**: 55-65%
- **Profit Factor Target**: > 1.5
- **Max Drawdown**: < 15%

### Best Practices

1. ✅ Always start with testnet
2. ✅ Backtest your settings thoroughly
3. ✅ Use stop losses
4. ✅ Don't over-leverage
5. ✅ Monitor funding rates
6. ✅ Keep detailed logs
7. ✅ Start small, scale gradually

---

<div align="center">

**Made with ❤️ for algorithmic traders**

⭐ **If you find this useful, please star the repo!** ⭐

[Report Bug](https://github.com/yourusername/binance-bot/issues) · 
[Request Feature](https://github.com/yourusername/binance-bot/issues)

</div>

---

**Version**: 6.0.0  
**Status**: Production-Ready ✅  
**Last Updated**: October 9, 2025
