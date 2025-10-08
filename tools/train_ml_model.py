#!/usr/bin/env python3
"""
Train ML Model for Signal Enhancement
Uses historical trades to learn pattern biases and improve confidence scoring
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ml_model import MLSignalEnhancer
from strategy import Strategy
from config import Config
from indicators import TechnicalIndicators
from binance_futures import BinanceFutures
from dotenv import load_dotenv

load_dotenv()


def generate_synthetic_training_data(num_samples=400):
    """
    Generate synthetic training data for demonstration
    In production, use real historical data
    """
    print("\n📊 Generating synthetic training data...")
    print("   (In production, use real Binance historical data)")
    print()
    
    # Create sample price data with more volatility
    base_price = 40000
    dates = [datetime.now() - timedelta(minutes=5*i) for i in range(num_samples)]
    dates.reverse()
    
    # Mix of strong trends and volatility
    trend = np.linspace(0, 3000, num_samples)
    cycle = np.sin(np.linspace(0, 6*np.pi, num_samples)) * 800  # More volatile
    noise = np.random.randn(num_samples) * 250  # More noise
    
    prices = base_price + trend + cycle + noise
    
    data = []
    for i, price in enumerate(prices):
        high = price + abs(np.random.randn() * 40)
        low = price - abs(np.random.randn() * 40)
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = abs(np.random.randn() * 100 + 500)
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # Calculate indicators
    df = TechnicalIndicators.add_all_indicators(df)
    
    print(f"✓ Generated {len(df)} candles with indicators")
    return df


def generate_signals_and_outcomes(df, strategy):
    """
    Generate signals for each point and simulate outcomes
    For training purposes, we'll create synthetic signals based on indicators
    """
    print("\n🔄 Generating training signals...")
    
    signals = []
    outcomes = []
    
    # Generate training samples from indicator patterns
    for i in range(150, len(df) - 10, 2):  # Every 2nd candle
        current = df.iloc[i]
        df_slice = df.iloc[:i+1]
        
        # Create synthetic signals based on indicator patterns
        signal_dict = None
        
        # Get indicators (with None checks)
        rsi = current.get('rsi')
        macd_hist = current.get('macd_hist')
        adx = current.get('adx')
        stochrsi_k = current.get('stochrsi_k')
        bb_position = ((current['close'] - current.get('bb_lower', current['close'])) / 
                      (current.get('bb_upper', current['close']) - current.get('bb_lower', current['close'])) 
                      if current.get('bb_upper') and current.get('bb_lower') and 
                      (current.get('bb_upper') - current.get('bb_lower')) > 0 else 0.5)
        
        # Skip if critical indicators are NaN
        if pd.isna(rsi) or pd.isna(macd_hist):
            continue
        
        # Generate signals from patterns
        # Bullish patterns
        if rsi < 40 and macd_hist > 0:
            signal_dict = {
                'signal': 'BUY',
                'confidence': 55.0 + np.random.rand() * 15,  # 55-70%
                'reasons': ['Oversold + MACD bullish'],
                'triggered_rules': ['Mean Reversion']
            }
        elif rsi > 60 and macd_hist < 0:
            signal_dict = {
                'signal': 'SELL',
                'confidence': 55.0 + np.random.rand() * 15,
                'reasons': ['Overbought + MACD bearish'],
                'triggered_rules': ['Mean Reversion']
            }
        # Add trend following signals
        elif not pd.isna(adx) and adx > 25 and macd_hist > 0.5:
            signal_dict = {
                'signal': 'BUY',
                'confidence': 60.0 + np.random.rand() * 20,
                'reasons': ['Strong uptrend'],
                'triggered_rules': ['Trend Following']
            }
        elif not pd.isna(adx) and adx > 25 and macd_hist < -0.5:
            signal_dict = {
                'signal': 'SELL',
                'confidence': 60.0 + np.random.rand() * 20,
                'reasons': ['Strong downtrend'],
                'triggered_rules': ['Trend Following']
            }
        
        if signal_dict is None:
            continue
        
        # Simulate outcome: check if price moved in expected direction
        future_prices = df.iloc[i:i+10]['close'].values
        entry_price = current['close']
        
        if len(future_prices) < 5:
            continue
        
        # Check profitability
        if signal_dict['signal'] == 'BUY':
            exit_price = max(future_prices)
            profit_pct = ((exit_price - entry_price) / entry_price) * 100
        else:  # SELL
            exit_price = min(future_prices)
            profit_pct = ((entry_price - exit_price) / entry_price) * 100
        
        # Outcome
        outcome = 1 if profit_pct > 0.5 else 0
        
        signals.append(signal_dict)
        outcomes.append(outcome)
    
    print(f"✓ Generated {len(signals)} training samples")
    if len(outcomes) > 0:
        win_rate = sum(outcomes) / len(outcomes) * 100
        print(f"   Win rate: {sum(outcomes)}/{len(outcomes)} ({win_rate:.1f}%)")
    
    return signals, outcomes


def fetch_real_binance_data(symbol: str, interval: str, limit: int = 1000):
    """
    Fetch real historical kline data from Binance
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Timeframe (e.g., '5m', '15m', '1h')
        limit: Number of candles (max 1500)
    """
    print(f"\n📈 Fetching real Binance data...")
    print(f"   Symbol: {symbol}")
    print(f"   Interval: {interval}")
    print(f"   Candles: {limit}")
    print()
    
    try:
        # Get API credentials
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        testnet = os.getenv('TESTNET', 'true').lower() == 'true'
        
        if not api_key or not api_secret:
            print("⚠ No API credentials - using public data")
            binance = BinanceFutures('', '', testnet)
        else:
            binance = BinanceFutures(api_key, api_secret, testnet)
        
        # Fetch klines
        df = binance.get_klines(symbol, interval, limit=limit)
        
        print(f"✓ Fetched {len(df)} real market candles")
        print(f"   Period: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print()
        
        # Calculate indicators
        print("🔄 Calculating 45+ technical indicators...")
        df = TechnicalIndicators.add_all_indicators(df)
        print("✓ Indicators calculated")
        print()
        
        return df
        
    except Exception as e:
        print(f"❌ Error fetching Binance data: {e}")
        return None


def main():
    """Main training function"""
    print("=" * 70)
    print("ML MODEL TRAINING FOR HYBRID STRATEGY")
    print("=" * 70)
    print()
    print("This script trains an ML model to enhance Experta rule-based signals.")
    print()
    print("🧠 Hybrid Approach:")
    print("   1. Experta rules → Base signal (explainable)")
    print("   2. ML model → Confidence adjustment (pattern learning)")
    print("   3. Final signal → 70% rules + 30% ML")
    print()
    
    # Create config and strategy
    config = Config()
    
    # Get symbol from user
    symbol = input("Trading symbol [BTCUSDT]: ").strip().upper() or 'BTCUSDT'
    interval = input("Timeframe [5m]: ").strip().lower() or '5m'
    
    config.trading_pair = symbol
    experta_strategy = Strategy(config)
    
    # Choose data source
    print("\n" + "=" * 70)
    print("DATA SOURCE")
    print("=" * 70)
    print("1. Real Binance kline data (RECOMMENDED) 🌟")
    print("2. Synthetic data (for testing)")
    print()
    
    data_choice = input("Select [1]: ").strip() or '1'
    
    if data_choice == '1':
        # Option 1: Real Binance data
        print("\n" + "=" * 70)
        print("FETCHING REAL BINANCE DATA")
        print("=" * 70)
        
        df = fetch_real_binance_data(symbol, interval, limit=1000)
        
        if df is None:
            print("❌ Failed to fetch data. Exiting.")
            return
    else:
        # Option 2: Synthetic data
        print("\n" + "=" * 70)
        print("GENERATING SYNTHETIC DATA")
        print("=" * 70)
        df = generate_synthetic_training_data(num_samples=400)
    
    # Generate signals and outcomes
    choice = 'y'
    
    # Generate signals and outcomes
    signals, outcomes = generate_signals_and_outcomes(df, experta_strategy)
    
    if len(signals) < 30:
        print("\n❌ Not enough signals generated for training")
        print(f"   Generated: {len(signals)}")
        print(f"   Needed: 30+")
        print()
        print("💡 Try:")
        print("   - Longer history (more candles)")
        print("   - Different timeframe")
        print("   - More volatile market data")
        return
    
    # Train ML model
    print("\n" + "=" * 70)
    print("TRAINING ML MODEL")
    print("=" * 70)
    print(f"   Target: {symbol} {interval}")
    print()
    
    # Create symbol-specific ML enhancer
    ml_enhancer = MLSignalEnhancer(symbol=symbol, timeframe=interval)
    success = ml_enhancer.train(df, signals, outcomes)
    
    if success:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print()
        
        # Show model info
        model_info = ml_enhancer.get_model_info()
        print(f"✅ Model trained for: {model_info['symbol']} {model_info['timeframe']}")
        print(f"✅ Saved to: {model_info['model_path']}")
        print()
        
        # Show feature importance
        importance = ml_enhancer.get_feature_importance()
        print("📊 Top 5 Most Important Features:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (feature, score) in enumerate(sorted_features, 1):
            print(f"   {i}. {feature}: {score:.3f}")
        print()
        
        print("🚀 Next steps:")
        print(f"   1. Run bot with: python main.py")
        print(f"      - Set TRADING_PAIR={symbol} in .env")
        print(f"      - Set TIMEFRAME={interval} in .env")
        print(f"   2. Bot will auto-load this model for {symbol} {interval}")
        print(f"   3. Train other symbols/timeframes:")
        print(f"      - ETHUSDT 5m, BTCUSDT 15m, etc.")
        print()
        
        # List all trained models
        import glob
        all_models = glob.glob('models/ml_*.pkl')
        if len(all_models) > 0:
            print("📦 All Trained Models:")
            for model_path in sorted(all_models):
                model_name = os.path.basename(model_path)
                # Extract symbol and timeframe from filename
                parts = model_name.replace('ml_', '').replace('.pkl', '').split('_')
                if len(parts) >= 2:
                    print(f"   ✓ {parts[0]} {parts[1]}")
            print()
        
        print("=" * 70)
        print()
    else:
        print("\n❌ Training failed")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

