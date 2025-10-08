#!/usr/bin/env python3
"""
Test Hybrid Strategy (Experta + ML)
Shows how ML enhances rule-based signals
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance_futures import BinanceFutures
from strategy import Strategy
from ml_model import HybridStrategy, MLSignalEnhancer
from indicators import TechnicalIndicators
from config import Config
from dotenv import load_dotenv

load_dotenv()


def main():
    print("=" * 70)
    print("HYBRID STRATEGY TEST (Experta + ML)")
    print("=" * 70)
    print()
    
    # Create config
    config = Config()
    symbol = input("Trading symbol [BTCUSDT]: ").strip().upper() or 'BTCUSDT'
    interval = input("Timeframe [5m]: ").strip().lower() or '5m'
    
    config.trading_pair = symbol
    
    print(f"\n🔍 Fetching current market data...")
    
    try:
        # Get API credentials
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        testnet = os.getenv('TESTNET', 'true').lower() == 'true'
        
        binance = BinanceFutures(api_key, api_secret, testnet)
        
        # Fetch current data
        df = binance.get_klines(symbol, interval, limit=200)
        print(f"✓ Fetched {len(df)} candles")
        
        # Calculate indicators
        df = TechnicalIndicators.add_all_indicators(df)
        print(f"✓ Indicators calculated")
        print()
        
        # Create strategies
        print("=" * 70)
        print("COMPARING: RULES ONLY vs HYBRID (RULES + ML)")
        print("=" * 70)
        print()
        
        # Strategy 1: Rules only (Experta)
        experta_only = Strategy(config)
        
        # Strategy 2: Hybrid (Experta + ML)
        ml_enhancer = MLSignalEnhancer()
        hybrid = HybridStrategy(config, experta_only, ml_enhancer)
        
        # Get signals
        print("🔍 Analyzing current market...\n")
        
        # Experta-only signal
        print("📊 1. EXPERTA RULES ONLY:")
        print("-" * 70)
        experta_signal = experta_only.analyze(df)
        
        print(f"Signal: {experta_signal['signal']}")
        print(f"Confidence: {experta_signal['confidence']:.1f}%")
        print(f"Reasons:")
        for reason in experta_signal['reasons'][:5]:
            print(f"  • {reason}")
        
        if 'triggered_rules' in experta_signal:
            print(f"\nTriggered Rules ({len(experta_signal['triggered_rules'])}):")
            for rule in experta_signal['triggered_rules'][:5]:
                print(f"  • {rule}")
        print()
        
        # Hybrid signal
        print("🤖 2. HYBRID (EXPERTA + ML):")
        print("-" * 70)
        
        # Create fresh strategy instance for hybrid
        experta_for_hybrid = Strategy(config)
        hybrid = HybridStrategy(config, experta_for_hybrid, ml_enhancer)
        
        hybrid_signal = hybrid.analyze(df)
        
        print(f"Signal: {hybrid_signal['signal']}")
        print(f"Confidence: {hybrid_signal['confidence']:.1f}%")
        
        if 'ml_adjustment' in hybrid_signal:
            ml_adj = hybrid_signal['ml_adjustment']
            ml_conf = hybrid_signal['ml_confidence']
            
            if ml_adj > 0:
                print(f"ML Boost: +{ml_adj:.1f}% (ML confidence: {ml_conf:.1f}%)")
            elif ml_adj < 0:
                print(f"ML Caution: {ml_adj:.1f}% (ML confidence: {ml_conf:.1f}%)")
            else:
                print(f"ML Neutral: {ml_adj:.1f}%")
        
        print(f"\nReasons:")
        for reason in hybrid_signal['reasons'][:5]:
            print(f"  • {reason}")
        print()
        
        # Comparison
        print("=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print()
        
        conf_diff = hybrid_signal['confidence'] - experta_signal['confidence']
        
        print(f"Experta Only:  {experta_signal['confidence']:.1f}%")
        print(f"Hybrid (ML):   {hybrid_signal['confidence']:.1f}%")
        print(f"Difference:    {conf_diff:+.1f}%")
        print()
        
        if ml_enhancer.is_trained:
            print("✅ ML model is active and enhancing signals!")
            print()
            print("💡 What ML adds:")
            print("   • Pattern recognition from 1000 candles of real data")
            print("   • Bias correction based on learned outcomes")
            print("   • Confidence adjustment (70% rules + 30% ML)")
            print("   • Top features: CMF, ADX, Trend Score")
        else:
            print("⚠️  ML model not trained - using rules only")
            print("   Train with: python tools/train_ml_model.py")
        
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

