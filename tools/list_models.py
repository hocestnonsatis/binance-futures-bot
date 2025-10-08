#!/usr/bin/env python3
"""
List all trained ML models
Shows which symbols/timeframes have trained models
"""

import os
import glob
from datetime import datetime


def list_models():
    """List all trained ML models"""
    print("=" * 70)
    print("TRAINED ML MODELS")
    print("=" * 70)
    print()
    
    # Find all ML models
    models_pattern = os.path.join('models', 'ml_*.pkl')
    all_models = glob.glob(models_pattern)
    
    if not all_models:
        print("⚠ No trained models found")
        print()
        print("Train a model with:")
        print("  python tools/train_ml_model.py")
        print()
        return
    
    print(f"Found {len(all_models)} trained model(s):\n")
    
    for model_path in sorted(all_models):
        model_name = os.path.basename(model_path)
        
        # Extract symbol and timeframe
        parts = model_name.replace('ml_', '').replace('.pkl', '').split('_')
        
        if len(parts) >= 2:
            symbol = parts[0]
            timeframe = parts[1]
            
            # Get file size and modification time
            file_size = os.path.getsize(model_path) / 1024  # KB
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
            
            print(f"✓ {symbol:12} {timeframe:6}  |  Size: {file_size:6.1f} KB  |  Updated: {mod_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            # Legacy format
            print(f"⚠ {model_name} (old format)")
    
    print()
    print("=" * 70)
    print()
    print("💡 Usage:")
    print("  1. Set TRADING_PAIR and TIMEFRAME in .env")
    print("  2. Run: python main.py")
    print("  3. Bot will auto-load the correct model")
    print()
    print("🎯 Train new models:")
    print("  python tools/train_ml_model.py")
    print()


if __name__ == '__main__':
    list_models()

