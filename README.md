# Binance Futures Trading Bot

A professional automated trading bot for Binance Futures with **Hybrid AI** (Expert System + Machine Learning), automatic model training, and comprehensive risk management.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)]()

---

## 🚀 Features

### Core Features

- **🤖 Hybrid AI Strategy**: Combines Experta rule-based system (explainable) with ML pattern recognition (powerful)
- **⚡ Auto-Training**: Automatically trains ML models on first run using real Binance data (30-60 seconds)
- **🎯 Multi-Symbol/Timeframe**: Separate trained models for each symbol-timeframe combination
- **📊 45+ Technical Indicators**: Powered by pandas_ta - RSI, MACD, EMA, ADX, Supertrend, Ichimoku, and more
- **🧠 Expert System**: 40+ intelligent trading rules with priority-based execution
- **📈 Backtesting Engine**: Test strategies on historical data with detailed performance metrics
- **🛡️ Risk Management**: Position sizing, stop-loss, take-profit, trailing stops, daily loss limits
- **💾 Database Logging**: SQLite for tracking trades and bot activity
- **🔄 Real-time Trading**: REST API polling for stable, reliable data

### Advanced Features

- **Market Regime Detection**: Automatically adapts to TRENDING, RANGING, or VOLATILE conditions
- **Explainable AI**: Every signal shows which rules triggered and why
- **ML Confidence Adjustment**: 70% rules + 30% ML for optimal decision-making
- **Funding Rate Tracking**: Monitor Binance Futures holding costs
- **Position Monitoring**: Real-time P&L tracking with trailing stops
- **Production-Ready**: Clean, modular, well-documented code

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Hybrid AI Strategy](#hybrid-ai-strategy)
- [Auto-Training](#auto-training)
- [Backtesting](#backtesting)
- [Risk Management](#risk-management)
- [Tools & Utilities](#tools--utilities)
- [Project Structure](#project-structure)
- [Future Roadmap](#future-roadmap)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/hocestnonsatis/binance-futures-bot.git
cd binance-futures-bot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your Binance API keys

# 5. Run the bot
python main.py

# On first run, bot will offer to auto-train ML model
# Just press [Y] and wait 30-60 seconds!
```

**That's it!** 🎉 Your bot is ready to trade with ML enhancement.

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
POSITION_SIZE_PCT=90  # Use 90% of available balance
STOP_LOSS_PCT=2.0     # Stop loss at -2%
TAKE_PROFIT_PCT=5.0   # Take profit at +5%
TRAILING_STOP_PCT=1.5 # Trailing stop distance
MAX_DAILY_LOSS_PCT=5.0 # Max daily loss before stopping

# Signal Settings
MIN_CONFIDENCE=55  # Minimum signal confidence (0-100)
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

## 🎮 Usage

### Running the Bot

```bash
python main.py
```

### First Run - Auto-Training

On the first run with a new symbol/timeframe, the bot will offer to train an ML model:

```
✓ Connected to Binance TESTNET
✓ Strategy: Hybrid AI Strategy (Experta + ML) [Rules-Only]
  🤖 ML Enhancement: Not trained
     No model for: BTCUSDT 5m

Would you like to train an ML model now?
This will fetch 1000 candles from Binance and train a model (~30-60 seconds)
Auto-train ML model? [Y/n]: 
```

Press **Y** and the bot will:
1. Fetch 1000 historical candles from Binance
2. Calculate 45+ technical indicators
3. Generate ~180-260 training samples
4. Train a Gradient Boosting model
5. Save the model for future use

**Next time you run the bot**, the model loads instantly! ⚡

### Subsequent Runs

```
✓ Connected to Binance TESTNET
✓ ML model loaded: BTCUSDT 5m
✓ Strategy: Hybrid AI Strategy (Experta + ML) [ML-Enhanced]
  🤖 ML Enhancement: Active
     Model: BTCUSDT 5m

Bot Ready!
```

The bot is now monitoring the market and will execute trades based on signals.

### Signal Output

```
[2025-10-08 14:32:15] Signal: BUY (72.5%)
  Reasons:
    - Perfect uptrend: ADX+EMA+DMP+Supertrend
    - ML boost: +7.5% (pattern recognition)
    - Volume confirms: CMF > 0.15

Triggered Rules:
  - Strong Trending Market
  - Perfect Uptrend
  - Volume Confirmation

Opening LONG position...
✓ Order placed: 0.05 BTC @ $42,350.00
```

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
│  │  • Priority-based    │     │  • 16 features       │    │
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
3. **Market Regime Detection**: Identify current market conditions
4. **Signal Generation**: 
   - Experta rules fire based on indicators (70% weight)
   - ML model predicts signal success (30% weight)
   - Combine for final confidence
5. **Risk Check**: Verify signal meets confidence threshold
6. **Position Sizing**: Calculate position size based on risk parameters
7. **Order Execution**: Place order on Binance
8. **Position Management**: Monitor with trailing stops, TP/SL

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

1. **Market Regime Detection (3 rules)**
   - Strong Trending Market
   - Ranging Market
   - Volatile Trending Market

2. **Trend Following (4 rules)**
   - Perfect Uptrend (ADX + EMA + DMP + Supertrend)
   - Perfect Downtrend
   - Bullish/Bearish Momentum Confirmation

3. **Mean Reversion (4 rules)**
   - Extreme Oversold/Overbought
   - Mean Reversion from EMA21

4. **Breakout Detection (4 rules)**
   - Bullish/Bearish Breakouts
   - Supertrend Reversals

5. **Volume Confirmation (2 rules)**
   - CMF + OBV validation

6. **Risk Filters (2 rules)**
   - Block trades in extreme conditions

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

**Pattern recognition and bias correction.**

- **Gradient Boosting Classifier** (scikit-learn)
- **16 Features**: RSI, ADX, MACD, CMF, MFI, trend score, etc.
- **Learns from 1000+ real market candles**
- **Symbol-specific models** (BTCUSDT ≠ ETHUSDT)
- **Confidence adjustment**: Boosts or reduces signal strength

**ML Features:**
```
1. rule_signal         - BUY/SELL/HOLD
2. rule_confidence     - Base confidence
3. num_rules          - Rules triggered
4. rsi, adx, cci      - Momentum indicators
5. macd_hist          - Trend strength
6. cmf, mfi           - Volume indicators
7. bb_position        - Bollinger Band position
8. trend_score        - Composite trend
9. ema_alignment      - EMA stack
10. recent_momentum    - Price action
... and 6 more
```

**Training:**
- **Data**: 1000 candles from Binance
- **Samples**: ~180-260 training examples
- **Accuracy**: 74-84% on test set
- **Time**: 30-60 seconds to train

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

## ⚡ Auto-Training

### Zero Manual Configuration

The bot includes **automatic ML model training**:

- **First run**: Bot detects missing model and offers to train
- **User choice**: Press [Y] to auto-train, [N] to use rules only
- **Fast training**: 30-60 seconds using real Binance data
- **Automatic loading**: Next run loads model instantly

### How Auto-Training Works

```
1. Fetch Historical Data
   ↓ 1000 candles from Binance (~5-10 sec)
   
2. Calculate Indicators
   ↓ 45+ technical indicators (~5-10 sec)
   
3. Generate Training Samples
   ↓ ~180-260 signal examples (~5-10 sec)
   
4. Train ML Model
   ↓ Gradient Boosting Classifier (~10-20 sec)
   
5. Save Model
   ↓ models/ml_{SYMBOL}_{TIMEFRAME}.pkl

Total: 30-60 seconds
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

### Managing Models

```bash
# List all trained models
python tools/list_models.py

# Output:
# ✓ BTCUSDT  5m   | Size: 231.9 KB | Updated: 2025-10-08 17:35
# ✓ ETHUSDT  15m  | Size: 298.7 KB | Updated: 2025-10-08 17:40

# Delete a model (will auto-train on next run)
rm models/ml_BTCUSDT_5m.pkl
```

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

### Funding Rate Awareness

Binance Futures charges funding rates every 8 hours. The bot:
- ✅ Tracks funding rates
- ✅ Logs costs in database
- ✅ Considers in P&L calculations

**Check current funding rate:**
```bash
python tools/test_funding_rate.py
```

---

## 🔧 Tools & Utilities

### Training & Testing

```bash
# Manual ML model training (advanced)
python tools/train_ml_model.py

# Test hybrid vs rules-only strategy
python tools/test_hybrid_strategy.py

# List all trained models
python tools/list_models.py

# Test indicators calculation
python tools/test_indicators.py
```

### Monitoring

```bash
# Check open position status
python tools/check_position.py

# Output:
# Position: LONG BTCUSDT
# Entry: $42,350.00
# Current: $42,850.00
# P&L: +$125.00 (+1.18%)
# Trailing Stop: $42,600.00

# Check funding rates
python tools/test_funding_rate.py
```

### Backtesting

```bash
# Backtest strategy on historical data
python tools/backtest.py
```

---

## 📁 Project Structure

```
binance-futures-bot/
├── Core (8 files)
│   ├── main.py                 # Bot main loop
│   ├── strategy.py             # Experta expert system (40+ rules)
│   ├── ml_model.py             # ML enhancement + auto-training
│   ├── indicators.py           # 45+ technical indicators
│   ├── binance_futures.py      # Binance API wrapper
│   ├── config.py               # Configuration management
│   ├── database.py             # SQLite database
│   └── risk_manager.py         # Risk management
│
├── models/                      # Trained ML models
│   ├── ml_BTCUSDT_5m.pkl
│   └── ...
│
├── tools/                       # Utility scripts
│   ├── train_ml_model.py       # Manual training
│   ├── backtest.py             # Backtesting engine
│   ├── list_models.py          # Model management
│   ├── test_hybrid_strategy.py # Strategy comparison
│   ├── test_indicators.py      # Indicator testing
│   ├── test_funding_rate.py    # Funding rate checker
│   └── check_position.py       # Position monitor
│
├── .env                         # Configuration (create from .env.example)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── LICENSE                      # MIT License
```

**Total:** ~8,000 lines of production-ready code

---

## 🔮 Future Roadmap

### v5.3 - Silent Auto-Training (Next)

- **Background model training**: Train models without user interaction
- **Scheduled retraining**: Auto-retrain models weekly/monthly
- **Model versioning**: Keep multiple model versions

### v6.0 - Advanced ML

- **LSTM/Transformer integration**: Deep learning for time-series prediction
- **Multi-timeframe analysis**: Combine 5m + 15m + 1h signals
- **Ensemble models**: XGBoost + LightGBM + CatBoost voting

### v6.5 - Online Learning

- **Incremental updates**: Model learns from each trade
- **Continuous adaptation**: No batch retraining needed
- **Reinforcement learning**: Q-learning for optimal actions

### v7.0 - Enhanced UX

- **Web dashboard**: Real-time monitoring and control
- **Telegram bot**: Notifications and remote control
- **Performance analytics**: Detailed charts and reports

### v8.0 - Advanced Features

- **Multi-symbol trading**: Trade multiple pairs simultaneously
- **Portfolio optimization**: Allocation across symbols
- **Genetic algorithm**: Auto-tune rule weights
- **Multi-exchange support**: Bybit, OKX, etc.

---

## ❓ FAQ

### General

**Q: Is this profitable?**  
A: Past performance doesn't guarantee future results. Backtest thoroughly and start with small amounts on testnet.

**Q: Do I need ML experience?**  
A: No! Auto-training handles everything. Just run the bot and press [Y].

**Q: Can I use this on live trading?**  
A: Yes, but start with testnet first. Set `TESTNET=false` in `.env` when ready.

### Technical

**Q: How accurate is the ML model?**  
A: Typical test accuracy: 74-84%. Combined with rules, provides robust signals.

**Q: How often should I retrain models?**  
A: Weekly or after major market regime changes. Delete model and bot will auto-retrain.

**Q: Can I add my own rules?**  
A: Yes! Edit `strategy.py` and add new `@Rule` decorators.

**Q: What about funding rates?**  
A: Bot tracks them. On average, ~0.01% per 8 hours (~10% annually). Factor into strategy.

### Trading

**Q: What's the best timeframe?**  
A: 5m-15m for active trading, 1h-4h for swing trading. Backtest to find your preference.

**Q: How much capital do I need?**  
A: Minimum $100 on testnet for testing. $1000+ recommended for live trading.

**Q: Does it work on all pairs?**  
A: Yes, any Binance Futures pair. Each gets its own trained model.

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow existing code style
- Add tests for new features
- Update documentation
- Keep commits atomic and descriptive

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

- **Issues**: [GitHub Issues](https://github.com/hocestnonsatis/binance-futures-bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hocestnonsatis/binance-futures-bot/discussions)

---

## 🎯 Quick Reference

### Common Commands

```bash
# Run bot
python main.py

# Backtest
python tools/backtest.py

# Check position
python tools/check_position.py

# List models
python tools/list_models.py
```

### Key Metrics

- **Test Accuracy**: 74-84% (ML models)
- **Win Rate Target**: 55-65%
- **Profit Factor Target**: > 1.5
- **Max Drawdown**: < 15%

### Best Practices

1. ✅ Always start with testnet
2. ✅ Backtest your settings thoroughly
3. ✅ Use stop losses
4. ✅ Don't over-leverage
5. ✅ Monitor funding rates
6. ✅ Retrain models regularly
7. ✅ Keep detailed logs

---

<div align="center">

**Made with ❤️ by traders, for traders**

⭐ **If you find this useful, please star the repo!** ⭐

[Report Bug](https://github.com/hocestnonsatis/binance-futures-bot/issues) · 
[Request Feature](https://github.com/hocestnonsatis/binance-futures-bot/issues) · 
[Documentation](https://github.com/hocestnonsatis/binance-futures-bot/wiki)

</div>

---

**Version**: 5.2.0  
**Status**: Production-Ready ✅  
**Last Updated**: October 8, 2025
