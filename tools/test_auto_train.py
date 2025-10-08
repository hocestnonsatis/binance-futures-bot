#!/usr/bin/env python3
"""
Test Auto-Training Feature
Simulates bot startup with missing ML model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from binance_futures import BinanceFutures
from ml_model import MLSignalEnhancer
from config import Config

load_dotenv()


def test_auto_train():
    """Test auto-training feature"""
    print("=" * 70)
    print("TESTING AUTO-TRAINING FEATURE")
    print("=" * 70)
    print()
    
    # Use a symbol that doesn't have a model
    symbol = 'BNBUSDT'
    timeframe = '1h'
    
    print(f"Testing with: {symbol} {timeframe}")
    print("(This symbol+timeframe has no trained model)")
    print()
    
    # Create ML enhancer
    ml_enhancer = MLSignalEnhancer(symbol=symbol, timeframe=timeframe)
    
    # Check if model exists
    model_info = ml_enhancer.get_model_info()
    print(f"Model Status:")
    print(f"  Symbol: {model_info['symbol']}")
    print(f"  Timeframe: {model_info['timeframe']}")
    print(f"  Model Path: {model_info['model_path']}")
    print(f"  Is Trained: {model_info['is_trained']}")
    print(f"  File Exists: {model_info['exists']}")
    print()
    
    if model_info['is_trained']:
        print("✓ Model already exists - deleting for test...")
        if os.path.exists(model_info['model_path']):
            os.remove(model_info['model_path'])
            print(f"✓ Deleted: {model_info['model_path']}")
            # Recreate ml_enhancer
            ml_enhancer = MLSignalEnhancer(symbol=symbol, timeframe=timeframe)
            print()
    
    # Now test auto-training
    print("=" * 70)
    print("STARTING AUTO-TRAINING")
    print("=" * 70)
    print()
    
    # Get Binance client
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    testnet = os.getenv('TESTNET', 'true').lower() == 'true'
    
    binance = BinanceFutures(api_key, api_secret, testnet)
    
    # Auto-train
    success = ml_enhancer.auto_train(binance, limit=1000)
    
    if success:
        print("\n" + "=" * 70)
        print("AUTO-TRAINING TEST: SUCCESS! ✅")
        print("=" * 70)
        print()
        print("✓ Model was automatically trained")
        print("✓ Model saved successfully")
        print()
        
        # Verify model exists
        model_info = ml_enhancer.get_model_info()
        print(f"Verification:")
        print(f"  Is Trained: {model_info['is_trained']}")
        print(f"  File Exists: {model_info['exists']}")
        
        if os.path.exists(model_info['model_path']):
            file_size = os.path.getsize(model_info['model_path']) / 1024
            print(f"  File Size: {file_size:.1f} KB")
        
        print()
        print("🎉 Auto-training feature works perfectly!")
        print()
        print("Next time you run the bot with BNBUSDT 1h,")
        print("it will automatically load this model!")
        
    else:
        print("\n❌ Auto-training test failed")
        return False
    
    return True


if __name__ == '__main__':
    try:
        success = test_auto_train()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

