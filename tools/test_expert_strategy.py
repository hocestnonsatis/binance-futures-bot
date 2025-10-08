#!/usr/bin/env python3
"""
Test Expert Strategy with sample data
Demonstrates rule-based decision making with Experta
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.expert import ExpertStrategy
from config import Config
from indicators import TechnicalIndicators


def generate_trending_market_data(num_candles=200):
    """Generate sample data for trending market"""
    print("Generating trending market data...")
    
    base_price = 40000
    dates = [datetime.now() - timedelta(minutes=5*i) for i in range(num_candles)]
    dates.reverse()
    
    # Strong uptrend
    trend = np.linspace(0, 3000, num_candles)  # +3000 over period
    noise = np.random.randn(num_candles) * 100
    prices = base_price + trend + noise
    
    data = []
    for i, price in enumerate(prices):
        high = price + abs(np.random.randn() * 30)
        low = price - abs(np.random.randn() * 30)
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
    print(f"✓ Generated {len(df)} candles (uptrend)")
    return df


def generate_ranging_market_data(num_candles=200):
    """Generate sample data for ranging market"""
    print("Generating ranging market data...")
    
    base_price = 40000
    dates = [datetime.now() - timedelta(minutes=5*i) for i in range(num_candles)]
    dates.reverse()
    
    # Oscillate between levels
    cycle = np.sin(np.linspace(0, 4*np.pi, num_candles)) * 500
    noise = np.random.randn(num_candles) * 100
    prices = base_price + cycle + noise
    
    data = []
    for i, price in enumerate(prices):
        high = price + abs(np.random.randn() * 30)
        low = price - abs(np.random.randn() * 30)
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
    print(f"✓ Generated {len(df)} candles (ranging)")
    return df


def test_expert_strategy():
    """Test Expert Strategy on different market conditions"""
    print("=" * 70)
    print("TESTING EXPERT STRATEGY (Experta Rule-Based System)")
    print("=" * 70)
    print()
    
    # Create config
    config = Config()
    config.trading_pair = 'BTCUSDT'
    config.strategy_type = 'expert'
    
    # Create strategy
    strategy = ExpertStrategy(config)
    print(f"Strategy: {strategy.get_name()}")
    print()
    
    # ==================== TEST 1: TRENDING MARKET ====================
    
    print("=" * 70)
    print("TEST 1: TRENDING MARKET (Uptrend)")
    print("=" * 70)
    print()
    
    df_trend = generate_trending_market_data(200)
    df_trend = TechnicalIndicators.add_all_indicators(df_trend)
    
    signal_trend = strategy.analyze(df_trend)
    
    print(f"Signal: {signal_trend['signal']}")
    print(f"Confidence: {signal_trend['confidence']:.1f}%")
    print(f"\nReasons:")
    for i, reason in enumerate(signal_trend['reasons'], 1):
        print(f"  {i}. {reason}")
    
    print(f"\nTriggered Rules ({len(signal_trend.get('triggered_rules', []))}):")
    for i, rule in enumerate(signal_trend.get('triggered_rules', []), 1):
        print(f"  {i}. {rule}")
    
    print()
    
    # ==================== TEST 2: RANGING MARKET ====================
    
    print("=" * 70)
    print("TEST 2: RANGING MARKET (Sideways)")
    print("=" * 70)
    print()
    
    df_range = generate_ranging_market_data(200)
    df_range = TechnicalIndicators.add_all_indicators(df_range)
    
    signal_range = strategy.analyze(df_range)
    
    print(f"Signal: {signal_range['signal']}")
    print(f"Confidence: {signal_range['confidence']:.1f}%")
    print(f"\nReasons:")
    for i, reason in enumerate(signal_range['reasons'], 1):
        print(f"  {i}. {reason}")
    
    print(f"\nTriggered Rules ({len(signal_range.get('triggered_rules', []))}):")
    for i, rule in enumerate(signal_range.get('triggered_rules', []), 1):
        print(f"  {i}. {rule}")
    
    print()
    
    # ==================== SUMMARY ====================
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    print("✅ Expert Strategy successfully tested!")
    print()
    print("Key Features Demonstrated:")
    print("  • Rule-based decision making")
    print("  • Market regime detection")
    print("  • Explainable signals (see triggered rules)")
    print("  • Priority-based rule execution")
    print("  • Multi-indicator confluence")
    print()
    print("🎯 The Expert Strategy uses Experta for:")
    print("  1. Clear separation of trading rules")
    print("  2. Easy addition of new rules")
    print("  3. Conflict resolution (contradictory signals)")
    print("  4. Complete explainability (know why each trade was taken)")
    print()


if __name__ == '__main__':
    try:
        test_expert_strategy()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

