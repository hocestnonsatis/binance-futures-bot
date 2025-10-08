# Utility Tools

Bu klasörde bot'u test etmek ve izlemek için yardımcı scriptler bulunur.

## 📊 Test & Kontrol Scriptleri

### 1. **test_indicators.py**
Test all 45+ technical indicators:
```bash
python tools/test_indicators.py
```
- Verifies pandas_ta indicators work correctly
- Shows indicator values
- Displays market summary

### 2. **test_expert_strategy.py**
Test Expert System Strategy with Experta:
```bash
python tools/test_expert_strategy.py
```
- Tests rule-based decision making
- Shows triggered rules
- Demonstrates explainable AI

### 3. **test_funding_rate.py**
Check funding rates and holding costs:
```bash
python tools/test_funding_rate.py
```
- Current funding rate
- Historical rates
- Cost calculations (daily/weekly/monthly)
- Strategy recommendations

### 4. **check_position.py**
Check current open positions:
```bash
python tools/check_position.py
```
- Position details
- P&L status
- Exit levels (stop loss, take profit, trailing stop)
- Recent logs
- Analysis and recommendations

## 🚀 Quick Commands

```bash
# Activate venv first
cd /home/anil/projects/binance-bot
source venv/bin/activate

# Then run any tool
python tools/test_indicators.py
python tools/check_position.py
python tools/test_funding_rate.py
```

## 💡 When to Use

- **test_indicators.py**: After updating indicators.py or installing pandas_ta
- **test_expert_strategy.py**: When modifying expert.py or testing rules
- **test_funding_rate.py**: Before opening long-term positions
- **check_position.py**: When position seems stuck or behaving unexpectedly

## 📚 More Info

See main [README.md](../README.md) for full documentation.

