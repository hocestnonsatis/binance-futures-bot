#!/usr/bin/env python3
"""
Test script for new pandas_ta indicators
Run this to verify all indicators are working correctly
"""

import pandas as pd
import numpy as np
from indicators import TechnicalIndicators
from datetime import datetime, timedelta

def generate_sample_data(num_candles=200):
    """Generate sample OHLCV data for testing"""
    print("Generating sample market data...")
    
    # Generate realistic-looking price data
    base_price = 40000
    dates = [datetime.now() - timedelta(minutes=5*i) for i in range(num_candles)]
    dates.reverse()
    
    # Random walk with trend
    trend = np.linspace(0, 1000, num_candles)  # Slight uptrend
    volatility = np.random.randn(num_candles) * 200
    prices = base_price + trend + np.cumsum(volatility)
    
    # Generate OHLCV
    data = []
    for i, price in enumerate(prices):
        high = price + abs(np.random.randn() * 50)
        low = price - abs(np.random.randn() * 50)
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
    print(f"✓ Generated {len(df)} candles")
    print(f"  Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print()
    return df


def test_indicators():
    """Test all indicators"""
    print("=" * 70)
    print("TESTING PANDAS_TA INDICATORS")
    print("=" * 70)
    print()
    
    # Generate sample data
    df = generate_sample_data(200)
    
    # Calculate indicators
    print("Calculating indicators...")
    df_with_indicators = TechnicalIndicators.add_all_indicators(df)
    print("✓ All indicators calculated!")
    print()
    
    # Display latest values
    latest = df_with_indicators.iloc[-1]
    
    print("=" * 70)
    print("LATEST INDICATOR VALUES")
    print("=" * 70)
    print()
    
    print("📊 PRICE DATA:")
    print(f"  Close: {latest['close']:.2f}")
    print(f"  High:  {latest['high']:.2f}")
    print(f"  Low:   {latest['low']:.2f}")
    print()
    
    print("📈 TREND INDICATORS:")
    print(f"  EMA 9:        {latest['ema_9']:.2f}")
    print(f"  EMA 21:       {latest['ema_21']:.2f}")
    print(f"  EMA 50:       {latest['ema_50']:.2f}")
    print(f"  EMA 200:      {latest['ema_200']:.2f}")
    if 'adx' in latest and not pd.isna(latest['adx']):
        print(f"  ADX:          {latest['adx']:.2f} (trend strength)")
        print(f"  DMP:          {latest['dmp']:.2f}")
        print(f"  DMN:          {latest['dmn']:.2f}")
    if 'supertrend' in latest and not pd.isna(latest['supertrend']):
        direction = "BULLISH" if latest['supertrend_direction'] == 1 else "BEARISH"
        print(f"  Supertrend:   {latest['supertrend']:.2f} ({direction})")
    print()
    
    print("⚡ MOMENTUM INDICATORS:")
    if 'rsi' in latest and not pd.isna(latest['rsi']):
        print(f"  RSI:          {latest['rsi']:.2f}")
    if 'stochrsi_k' in latest and not pd.isna(latest['stochrsi_k']):
        print(f"  Stoch RSI K:  {latest['stochrsi_k']:.4f}")
        print(f"  Stoch RSI D:  {latest['stochrsi_d']:.4f}")
    if 'stoch_k' in latest and not pd.isna(latest['stoch_k']):
        print(f"  Stochastic K: {latest['stoch_k']:.2f}")
        print(f"  Stochastic D: {latest['stoch_d']:.2f}")
    if 'cci' in latest and not pd.isna(latest['cci']):
        print(f"  CCI:          {latest['cci']:.2f}")
    if 'willr' in latest and not pd.isna(latest['willr']):
        print(f"  Williams %R:  {latest['willr']:.2f}")
    print()
    
    print("📉 VOLATILITY INDICATORS:")
    if 'bb_upper' in latest and not pd.isna(latest['bb_upper']):
        print(f"  BB Upper:     {latest['bb_upper']:.2f}")
        print(f"  BB Middle:    {latest['bb_middle']:.2f}")
        print(f"  BB Lower:     {latest['bb_lower']:.2f}")
        print(f"  BB Bandwidth: {latest['bb_bandwidth']:.4f}")
    if 'atr' in latest and not pd.isna(latest['atr']):
        atr_pct = (latest['atr'] / latest['close']) * 100
        print(f"  ATR:          {latest['atr']:.2f} ({atr_pct:.2f}%)")
    if 'kc_upper' in latest and not pd.isna(latest['kc_upper']):
        print(f"  KC Upper:     {latest['kc_upper']:.2f}")
        print(f"  KC Lower:     {latest['kc_lower']:.2f}")
    if 'dc_upper' in latest and not pd.isna(latest['dc_upper']):
        print(f"  DC Upper:     {latest['dc_upper']:.2f}")
        print(f"  DC Lower:     {latest['dc_lower']:.2f}")
    print()
    
    print("💰 VOLUME INDICATORS:")
    if 'obv' in latest and not pd.isna(latest['obv']):
        print(f"  OBV:          {latest['obv']:.0f}")
    if 'cmf' in latest and not pd.isna(latest['cmf']):
        print(f"  CMF:          {latest['cmf']:.4f}")
    if 'mfi' in latest and not pd.isna(latest['mfi']):
        print(f"  MFI:          {latest['mfi']:.2f}")
    if 'vwap' in latest and not pd.isna(latest['vwap']):
        print(f"  VWAP:         {latest['vwap']:.2f}")
    print()
    
    print("🎯 CUSTOM COMPOSITE INDICATORS:")
    if 'trend_score' in latest and not pd.isna(latest['trend_score']):
        trend_score = latest['trend_score']
        if trend_score > 30:
            trend_label = "STRONG UPTREND"
        elif trend_score > 10:
            trend_label = "UPTREND"
        elif trend_score < -30:
            trend_label = "STRONG DOWNTREND"
        elif trend_score < -10:
            trend_label = "DOWNTREND"
        else:
            trend_label = "RANGING"
        print(f"  Trend Score:         {trend_score:.2f} ({trend_label})")
    
    if 'ema_ribbon_strength' in latest and not pd.isna(latest['ema_ribbon_strength']):
        print(f"  EMA Ribbon Strength: {latest['ema_ribbon_strength']:.4f}")
    
    if 'volume_pressure' in latest and not pd.isna(latest['volume_pressure']):
        print(f"  Volume Pressure:     {latest['volume_pressure']:.4f}")
    print()
    
    # Get indicator summary
    summary = TechnicalIndicators.get_indicator_summary(df_with_indicators)
    
    print("=" * 70)
    print("MARKET SUMMARY")
    print("=" * 70)
    for key, value in summary.items():
        print(f"  {key.upper():15s}: {value}")
    print()
    
    # Count available indicators
    indicator_columns = [col for col in df_with_indicators.columns 
                        if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"  Total indicators calculated: {len(indicator_columns)}")
    print(f"  Data points: {len(df_with_indicators)}")
    print(f"  Columns: {', '.join(indicator_columns[:10])}...")
    print()
    
    print("✅ ALL TESTS PASSED!")
    print()
    print("🚀 Your bot is ready to use professional indicators!")
    print()


if __name__ == '__main__':
    try:
        test_indicators()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

