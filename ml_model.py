"""
Machine Learning Model for Signal Enhancement
Hybrid AI: Experta Rules + ML Confidence Adjustment
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not available. Install with: pip install xgboost")


class MLSignalEnhancer:
    """
    ML Model to enhance Experta rule-based signals
    
    Architecture:
    1. Experta generates base signal (explainable, rule-based)
    2. ML model adjusts confidence based on learned patterns
    3. Final signal = Rule-based decision + ML confidence adjustment
    
    This provides:
    - Explainability (from rules)
    - Pattern recognition (from ML)
    - Bias correction (ML learns from mistakes)
    
    Multi-Symbol/Timeframe Support:
    - Each symbol+timeframe gets its own trained model
    - Example: BTCUSDT-5m, ETHUSDT-15m have separate models
    - Models automatically loaded based on current config
    """
    
    def __init__(self, symbol: str = None, timeframe: str = None, model_dir: str = 'models', 
                 alternative_symbol: str = None):
        """
        Initialize ML Signal Enhancer
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Candle timeframe (e.g., '5m', '15m', '1h')
            model_dir: Directory to store models
            alternative_symbol: Alternative symbol to use if primary model not found
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_dir = model_dir
        self.alternative_symbol = alternative_symbol
        self.actual_model_symbol = symbol  # Track which symbol's model is actually loaded
        
        # Create model filename based on symbol+timeframe
        if symbol and timeframe:
            model_filename = f"ml_{symbol}_{timeframe}.pkl"
        else:
            model_filename = "signal_enhancer.pkl"  # Legacy default
        
        self.model_path = os.path.join(model_dir, model_filename)
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Try to load existing model (with fallback to similar symbols)
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model if exists, with fallback to similar symbols"""
        # Try primary model
        if os.path.exists(self.model_path):
            try:
                saved_data = joblib.load(self.model_path)
                
                # Check feature version compatibility
                expected_feature_count = 27  # Current version has 27 features
                model_feature_count = saved_data.get('feature_count', None)
                model_version = saved_data.get('version', '1.0')
                
                if model_feature_count is not None and model_feature_count != expected_feature_count:
                    # Feature mismatch - model is outdated
                    print(f"⚠ Model incompatible: trained with {model_feature_count} features, need {expected_feature_count}")
                    print(f"  This model needs to be retrained with the updated features")
                    print(f"  Model version: {model_version} (current: 2.0)")
                    self.is_trained = False
                    return
                
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.is_trained = True
                self.actual_model_symbol = self.symbol
                
                if self.symbol and self.timeframe:
                    version_str = f" (v{model_version})" if model_version else ""
                    print(f"✓ ML model loaded: {self.symbol} {self.timeframe}{version_str}")
                else:
                    model_name = os.path.basename(self.model_path)
                    print(f"✓ ML model loaded: {model_name}")
                return
            except Exception as e:
                print(f"⚠ Could not load model: {e}")
                self.is_trained = False
                return
        
        # Primary model not found - try to find similar models
        if self.symbol and self.timeframe:
            similar_models = self._find_similar_models()
            
            if similar_models:
                print(f"\n⚠ No model found for {self.symbol} {self.timeframe}")
                print(f"   Found {len(similar_models)} similar model(s):\n")
                
                for idx, (sym, path) in enumerate(similar_models[:5], 1):
                    print(f"   {idx}. {sym} {self.timeframe}")
                
                # If alternative_symbol provided, use it automatically
                if self.alternative_symbol:
                    for sym, path in similar_models:
                        if sym == self.alternative_symbol:
                            self._load_alternative_model(path, sym)
                            return
                
                # Otherwise, use first similar model automatically for convenience
                # (user can always train a specific model later)
                auto_sym, auto_path = similar_models[0]
                print(f"\n   Using similar model: {auto_sym} {self.timeframe}")
                self._load_alternative_model(auto_path, auto_sym)
            else:
                self.is_trained = False
        else:
            self.is_trained = False
    
    def _find_similar_models(self) -> List[Tuple[str, str]]:
        """
        Find similar models based on symbol similarity
        
        Returns:
            List of (symbol, model_path) tuples
        """
        if not self.symbol or not self.timeframe:
            return []
        
        similar = []
        base_coin = self.symbol.replace('USDT', '').replace('USDC', '').replace('BUSD', '')
        
        # Check all model files
        for filename in os.listdir(self.model_dir):
            if not filename.endswith('.pkl') or not filename.startswith('ml_'):
                continue
            
            # Parse: ml_SYMBOL_TIMEFRAME.pkl
            parts = filename.replace('ml_', '').replace('.pkl', '').rsplit('_', 1)
            if len(parts) != 2:
                continue
            
            file_symbol, file_timeframe = parts
            
            # Only consider same timeframe
            if file_timeframe != self.timeframe:
                continue
            
            # Check similarity
            file_base_coin = file_symbol.replace('USDT', '').replace('USDC', '').replace('BUSD', '')
            
            # Same base coin but different stablecoin (SOLUSDT vs SOLUSDC)
            if file_base_coin == base_coin:
                model_path = os.path.join(self.model_dir, filename)
                similar.append((file_symbol, model_path))
        
        return similar
    
    def _load_alternative_model(self, model_path: str, symbol: str):
        """Load an alternative model with version checking"""
        try:
            saved_data = joblib.load(model_path)
            
            # Check feature version compatibility
            expected_feature_count = 27
            model_feature_count = saved_data.get('feature_count', None)
            model_version = saved_data.get('version', '1.0')
            
            if model_feature_count is not None and model_feature_count != expected_feature_count:
                print(f"⚠ Alternative model incompatible: {model_feature_count} features (need {expected_feature_count})")
                self.is_trained = False
                return
            
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.is_trained = True
            self.actual_model_symbol = symbol
            print(f"✓ Alternative model loaded: {symbol} → {self.symbol}")
        except Exception as e:
            print(f"⚠ Could not load alternative model: {e}")
            self.is_trained = False
    
    def get_model_info(self) -> Dict:
        """Get information about this model instance"""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'model_path': self.model_path,
            'is_trained': self.is_trained,
            'exists': os.path.exists(self.model_path),
            'actual_model_symbol': self.actual_model_symbol if self.is_trained else None,
            'is_alternative': self.is_trained and self.actual_model_symbol != self.symbol
        }
    
    def auto_train(self, binance_client=None, limit: int = 1000, df: pd.DataFrame = None) -> bool:
        """
        Automatically train model using Binance data or provided DataFrame
        
        Args:
            binance_client: BinanceFutures instance (optional if df provided)
            limit: Number of candles to fetch from Binance (default 1000)
            df: Pre-loaded DataFrame with OHLCV data (optional)
        
        Returns:
            True if training successful
        """
        if not self.symbol or not self.timeframe:
            print("❌ Cannot auto-train: Symbol and timeframe required")
            return False
        
        print(f"\n{'═' * 70}")
        print(f"AUTO-TRAINING ML MODEL: {self.symbol} {self.timeframe}")
        print(f"{'═' * 70}\n")
        
        try:
            # Import required modules
            from indicators import TechnicalIndicators
            from strategy import Strategy
            from config import Config
            from data.data_manager import DataManager
            from data.binance_downloader import BinanceDataDownloader
            import pandas as pd
            import numpy as np
            
            # Get data with priority: provided df > cached data > download from Binance
            if df is not None:
                # Use provided DataFrame
                print(f"📊 Using provided DataFrame: {len(df)} candles")
                if 'timestamp' not in df.columns:
                    print("❌ DataFrame must have 'timestamp' column")
                    return False
            else:
                # Try to load cached data first
                print(f"🔍 Checking for cached data: {self.symbol} {self.timeframe}...")
                data_manager = DataManager()
                df = data_manager.load_data(self.symbol, self.timeframe)
                
                if df is not None and len(df) >= 500:
                    # Use cached data (prefer more data for better training)
                    print(f"✅ Found cached data: {len(df)} candles")
                    print(f"   Period: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
                    # Use recent subset for training (last 2000 candles)
                    if len(df) > 2000:
                        df = df.tail(2000).reset_index(drop=True)
                        print(f"   Using most recent {len(df)} candles for training")
                elif binance_client is not None:
                    # No cached data or insufficient - download from Binance
                    print(f"📥 No cached data found, downloading from Binance...")
                    
                    # For auto-training, try to get more data than the API limit
                    # by using the downloader which handles pagination
                    try:
                        downloader = BinanceDataDownloader()
                        df = downloader.download_symbol(self.symbol, self.timeframe)
                        
                        if df is not None and len(df) > 0:
                            print(f"✅ Downloaded and cached {len(df)} candles")
                            # Use recent subset for training
                            if len(df) > 2000:
                                df = df.tail(2000).reset_index(drop=True)
                                print(f"   Using most recent {len(df)} candles for training")
                        else:
                            # Fallback to direct API call with limit
                            print(f"   Falling back to direct API call...")
                            actual_limit = min(limit, 1500)
                            df = binance_client.get_klines(self.symbol, self.timeframe, limit=actual_limit)
                    except Exception as e:
                        print(f"   Download failed: {e}")
                        print(f"   Falling back to direct API call...")
                        # Fallback to direct API call
                        actual_limit = min(limit, 1500)
                        df = binance_client.get_klines(self.symbol, self.timeframe, limit=actual_limit)
                else:
                    print("❌ No data source available (no cached data, no binance_client)")
                    return False
            
            # Check minimum data requirement
            if len(df) < 200:
                print(f"❌ Insufficient data: {len(df)} candles (need at least 200)")
                print("   This coin is too new or has low trading activity")
                return False
            
            print(f"✓ Fetched {len(df)} candles")
            print(f"   Period: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
            print()
            
            # Calculate indicators
            print("🔄 Calculating 45+ technical indicators...")
            df = TechnicalIndicators.add_all_indicators(df)
            print("✓ Indicators calculated")
            print()
            
            # Create strategy for signal generation
            config = Config()
            config.trading_pair = self.symbol
            config.timeframe = self.timeframe
            experta_strategy = Strategy(config)
            
            # Generate training samples with REAL price-based labels
            print("🔄 Generating training labels from price movement...")
            signals = []
            outcomes = []
            
            # Use strategy to generate realistic signals
            for i in range(150, len(df) - 20, 3):  # Every 3rd candle, leave 20 for outcome check
                current = df.iloc[i]
                df_slice = df.iloc[:i+1]
                
                # Get indicators
                rsi = current.get('rsi')
                macd_hist = current.get('macd_hist')
                adx = current.get('adx')
                trend_score = current.get('trend_score', 0)
                
                if pd.isna(rsi) or pd.isna(macd_hist):
                    continue
                
                # Generate signal using actual strategy patterns
                signal_dict = None
                
                # === Pattern 1: Strong Trend Following ===
                if trend_score > 30 and macd_hist > 0.2 and rsi > 45 and rsi < 70:
                    signal_dict = {
                        'signal': 'BUY',
                        'confidence': 65.0 + min(20, trend_score / 2),
                        'reasons': ['Strong uptrend with momentum'],
                        'triggered_rules': ['Trend Following']
                    }
                elif trend_score < -30 and macd_hist < -0.2 and rsi < 55 and rsi > 30:
                    signal_dict = {
                        'signal': 'SELL',
                        'confidence': 65.0 + min(20, abs(trend_score) / 2),
                        'reasons': ['Strong downtrend with momentum'],
                        'triggered_rules': ['Trend Following']
                    }
                
                # === Pattern 2: Mean Reversion in Range ===
                elif trend_score > -10 and trend_score < 10:  # Ranging
                    if rsi < 30 and current.get('stochrsi_k', 0.5) < 0.2:
                        signal_dict = {
                            'signal': 'BUY',
                            'confidence': 60.0,
                            'reasons': ['Oversold in range'],
                            'triggered_rules': ['Mean Reversion']
                        }
                    elif rsi > 70 and current.get('stochrsi_k', 0.5) > 0.8:
                        signal_dict = {
                            'signal': 'SELL',
                            'confidence': 60.0,
                            'reasons': ['Overbought in range'],
                            'triggered_rules': ['Mean Reversion']
                        }
                
                # === Pattern 3: Momentum Confirmation ===
                elif not pd.isna(adx) and adx > 25:
                    dmp = current.get('dmp', 0)
                    dmn = current.get('dmn', 0)
                    
                    if dmp > dmn and macd_hist > 0 and rsi > 50:
                        signal_dict = {
                            'signal': 'BUY',
                            'confidence': 62.0 + (adx - 25),
                            'reasons': ['Momentum breakout'],
                            'triggered_rules': ['Momentum']
                        }
                    elif dmn > dmp and macd_hist < 0 and rsi < 50:
                        signal_dict = {
                            'signal': 'SELL',
                            'confidence': 62.0 + (adx - 25),
                            'reasons': ['Momentum breakdown'],
                            'triggered_rules': ['Momentum']
                        }
                
                if signal_dict is None:
                    continue
                
                # === IMPROVED LABELING: Multi-threshold outcomes ===
                # Instead of fixed 1% target, use dynamic targets based on volatility
                future_slice = df.iloc[i+1:i+21]  # Next 20 candles
                entry_price = current['close']
                
                # Get ATR for dynamic targets
                atr = current.get('atr', entry_price * 0.02)
                atr_pct = (atr / entry_price) * 100
                
                # Dynamic profit target: 1.5x ATR or minimum 0.8%
                profit_target = max(0.008, atr_pct * 1.5 / 100)
                # Stop loss: 1x ATR or maximum 1.5%
                stop_loss = min(0.015, atr_pct / 100)
                
                if signal_dict['signal'] == 'BUY':
                    # Check if price reached profit target before stop loss
                    max_price = future_slice['close'].max()
                    min_price = future_slice['close'].min()
                    
                    profit_reached = (max_price - entry_price) / entry_price >= profit_target
                    stop_hit = (entry_price - min_price) / entry_price >= stop_loss
                    
                    # Outcome: 1 if profit reached first, 0 if stop hit or no clear winner
                    if profit_reached:
                        # Check if stop was hit first
                        for future_price in future_slice['close'].values:
                            if (future_price - entry_price) / entry_price >= profit_target:
                                outcome = 1  # Profit reached
                                break
                            if (entry_price - future_price) / entry_price >= stop_loss:
                                outcome = 0  # Stop hit first
                                break
                        else:
                            outcome = 1 if profit_reached else 0
                    else:
                        outcome = 0
                        
                else:  # SELL
                    # Check if price reached profit target before stop loss
                    max_price = future_slice['close'].max()
                    min_price = future_slice['close'].min()
                    
                    profit_reached = (entry_price - min_price) / entry_price >= profit_target
                    stop_hit = (max_price - entry_price) / entry_price >= stop_loss
                    
                    if profit_reached:
                        # Check order of events
                        for future_price in future_slice['close'].values:
                            if (entry_price - future_price) / entry_price >= profit_target:
                                outcome = 1  # Profit reached
                                break
                            if (future_price - entry_price) / entry_price >= stop_loss:
                                outcome = 0  # Stop hit first
                                break
                        else:
                            outcome = 1 if profit_reached else 0
                    else:
                        outcome = 0
                
                signals.append(signal_dict)
                outcomes.append(outcome)
            
            if len(signals) < 30:
                print(f"❌ Not enough signals generated: {len(signals)}/30")
                print("   Try with more candles or different timeframe")
                return False
            
            win_rate = (sum(outcomes) / len(outcomes) * 100) if len(outcomes) > 0 else 0
            print(f"✓ Generated {len(signals)} training samples")
            print(f"   Win rate: {sum(outcomes)}/{len(outcomes)} ({win_rate:.1f}%)")
            print()
            
            # Train model
            print("🤖 Training ML model...")
            success = self.train(df, signals, outcomes)
            
            if success:
                print(f"\n{'═' * 70}")
                print(f"AUTO-TRAINING COMPLETE!")
                print(f"{'═' * 70}")
                print(f"\n✅ Model ready: {self.symbol} {self.timeframe}")
                print(f"✅ Saved to: {self.model_path}")
                print()
                return True
            else:
                print("\n❌ Auto-training failed")
                return False
                
        except Exception as e:
            print(f"\n❌ Auto-training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_model(self):
        """Save trained model with feature metadata"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Define current feature list for version tracking
        feature_names = [
            'rule_signal', 'rule_confidence', 'num_rules',
            'rsi', 'adx', 'cci', 'macd_hist', 'stochrsi',
            'atr_pct', 'bb_position', 'cmf', 'mfi',
            'trend_score', 'ema_alignment', 'supertrend', 'recent_momentum',
            # Multi-timeframe features
            'htf_trend_alignment', 'htf_momentum_strength', 'htf_rsi_divergence',
            'htf_macd_confirmation', 'htf_strong_trend',
            # Interaction features
            'rsi_volume_interaction', 'macd_adx_interaction', 'bb_volatility_interaction',
            # Momentum features
            'rsi_momentum', 'macd_momentum', 'volume_momentum'
        ]
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_count': len(feature_names),
            'feature_names': feature_names,
            'version': '2.0'  # Feature version for compatibility
        }, self.model_path)
        print(f"✓ Model saved to {self.model_path}")
    
    def extract_features(self, df: pd.DataFrame, rule_signal: str, 
                        rule_confidence: float, triggered_rules: list) -> np.ndarray:
        """
        Extract features for ML model
        
        Enhanced with:
        - Technical indicator values
        - Rule-based signal information
        - Market regime indicators
        - Multi-timeframe features (if available)
        - Interaction features
        - Momentum features
        """
        latest = df.iloc[-1]
        
        features = []
        
        # === Rule-based features ===
        rule_signal_encoded = {'BUY': 1, 'SELL': -1, 'HOLD': 0}.get(rule_signal, 0)
        features.append(rule_signal_encoded)
        features.append(rule_confidence / 100.0)  # Normalize to 0-1
        features.append(len(triggered_rules))  # Number of rules triggered
        
        # === Technical indicators ===
        # Trend indicators
        rsi = latest.get('rsi', 50)
        adx = latest.get('adx', 20)
        cci = latest.get('cci', 0)
        
        features.append(rsi / 100.0)  # Normalize
        features.append(adx / 100.0)
        features.append(np.clip(cci, -200, 200) / 200.0)  # Normalize CCI
        
        # Momentum
        macd_hist = latest.get('macd_hist', 0)
        features.append(np.clip(macd_hist, -1, 1))  # Clip MACD histogram
        
        stochrsi_k = latest.get('stochrsi_k', 0.5)
        features.append(stochrsi_k)
        
        # Volatility
        atr = latest.get('atr', 0)
        atr_pct = (atr / latest['close']) * 100 if latest['close'] > 0 else 0
        features.append(np.clip(atr_pct, 0, 10) / 10.0)  # Normalize to 0-1
        
        # Bollinger position
        bb_upper = latest.get('bb_upper', latest['close'])
        bb_lower = latest.get('bb_lower', latest['close'])
        bb_position = (latest['close'] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
        features.append(bb_position)
        
        # Volume indicators
        cmf = latest.get('cmf', 0)
        features.append(np.clip(cmf, -1, 1))
        
        mfi = latest.get('mfi', 50)
        features.append(mfi / 100.0)
        
        # Trend score
        trend_score = latest.get('trend_score', 0)
        features.append(np.clip(trend_score, -100, 100) / 100.0)
        
        # EMA alignment
        ema_9 = latest.get('ema_9', latest['close'])
        ema_21 = latest.get('ema_21', latest['close'])
        ema_50 = latest.get('ema_50', latest['close'])
        
        ema_trend = 0
        if ema_9 > ema_21 > ema_50:
            ema_trend = 1  # Uptrend
        elif ema_9 < ema_21 < ema_50:
            ema_trend = -1  # Downtrend
        features.append(ema_trend)
        
        # Supertrend direction
        st_dir = latest.get('supertrend_direction', 0)
        features.append(st_dir if st_dir in [-1, 0, 1] else 0)
        
        # Recent price action (momentum)
        recent_returns = df['close'].pct_change().tail(5).mean()
        features.append(np.clip(recent_returns * 100, -5, 5) / 5.0)
        
        # === Multi-Timeframe Features (if available) ===
        # These features come from add_higher_timeframe_indicators()
        
        # HTF Trend Alignment
        htf_trend_alignment = latest.get('htf_trend_alignment', 0)
        features.append(np.clip(htf_trend_alignment, -2, 2) / 2.0)  # Normalize to -1 to 1
        
        # HTF Momentum Strength
        htf_momentum_strength = latest.get('htf_momentum_strength', 0)
        features.append(np.clip(htf_momentum_strength, -1, 1))
        
        # HTF RSI Divergence
        htf_rsi_div = latest.get('htf_rsi_divergence', 0)
        features.append(np.clip(htf_rsi_div, -50, 50) / 50.0)  # Normalize
        
        # HTF MACD confirmation
        htf_macd_bullish = latest.get('htf_macd_bullish', 0)
        htf_macd_bearish = latest.get('htf_macd_bearish', 0)
        features.append(htf_macd_bullish - htf_macd_bearish)  # -1, 0, or 1
        
        # HTF Strong Trend
        htf_strong_trend = latest.get('htf_strong_trend', 0)
        features.append(htf_strong_trend)
        
        # === Interaction Features ===
        # These capture non-linear relationships
        
        # RSI × Volume (strong momentum with volume confirmation)
        rsi_volume_interaction = (rsi / 100.0) * (mfi / 100.0)
        features.append(rsi_volume_interaction)
        
        # MACD × ADX (trend strength with momentum)
        macd_adx_interaction = np.clip(macd_hist, -1, 1) * (adx / 100.0)
        features.append(macd_adx_interaction)
        
        # BB Position × Volatility (overbought/oversold with volatility)
        bb_volatility_interaction = bb_position * (atr_pct / 10.0)
        features.append(bb_volatility_interaction)
        
        # === Momentum Features (Rate of Change) ===
        # Capture acceleration/deceleration
        
        # RSI momentum (is RSI rising or falling?)
        if len(df) >= 5:
            rsi_5_ago = df.iloc[-5].get('rsi', 50)
            rsi_momentum = (rsi - rsi_5_ago) / 5.0  # Change per candle
            features.append(np.clip(rsi_momentum, -10, 10) / 10.0)
        else:
            features.append(0)
        
        # MACD histogram momentum
        if len(df) >= 5:
            macd_hist_5_ago = df.iloc[-5].get('macd_hist', 0)
            macd_momentum = macd_hist - macd_hist_5_ago
            features.append(np.clip(macd_momentum, -1, 1))
        else:
            features.append(0)
        
        # Volume momentum (is volume increasing?)
        if len(df) >= 5:
            volume = latest.get('volume', 0)
            volume_5_ago = df.iloc[-5].get('volume', 0)
            if volume_5_ago > 0:
                volume_change = (volume - volume_5_ago) / volume_5_ago
                features.append(np.clip(volume_change, -2, 2) / 2.0)
            else:
                features.append(0)
        else:
            features.append(0)
        
        return np.array(features).reshape(1, -1)
    
    def train(self, df: pd.DataFrame, signals_history: list, outcomes: list):
        """
        Train ML model on historical signals and outcomes
        
        Args:
            df: Historical OHLCV data with indicators
            signals_history: List of (signal, confidence, rules) from Experta
            outcomes: List of trade outcomes (1 = profitable, 0 = loss)
        """
        if len(signals_history) < 50:
            print("⚠ Need at least 50 historical signals to train")
            return False
        
        print(f"\n🤖 Training ML Signal Enhancer...")
        print(f"   Training samples: {len(signals_history)}")
        
        # Extract features
        X = []
        y = []
        
        for i, (signal_dict, outcome) in enumerate(zip(signals_history, outcomes)):
            try:
                # Get the corresponding market data
                df_slice = df.iloc[:i+100]  # Use data up to that point
                
                if len(df_slice) < 100:
                    continue
                
                features = self.extract_features(
                    df_slice,
                    signal_dict['signal'],
                    signal_dict['confidence'],
                    signal_dict.get('triggered_rules', [])
                )
                
                X.append(features.flatten())
                y.append(outcome)
                
            except Exception as e:
                print(f"⚠ Error extracting features for sample {i}: {e}")
                continue
        
        if len(X) < 30:
            print("❌ Not enough valid samples to train")
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # === Enhanced Ensemble Model ===
        print("   Building ensemble model...")
        
        # Level 1: Base models
        base_models = []
        
        # Model 1: Gradient Boosting (best for trading patterns)
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
        base_models.append(('gradient_boosting', gb_model))
        print("     ✓ Gradient Boosting")
        
        # Model 2: Random Forest (diverse decision trees)
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=15,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42
        )
        base_models.append(('random_forest', rf_model))
        print("     ✓ Random Forest")
        
        # Model 3: XGBoost (if available, state-of-the-art gradient boosting)
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            base_models.append(('xgboost', xgb_model))
            print("     ✓ XGBoost")
        
        # Level 2: Meta-classifier (Stacking)
        # Logistic Regression combines predictions from base models
        meta_classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        
        # Create stacking ensemble
        print("   Training stacking ensemble...")
        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_classifier,
            cv=5,  # 5-fold cross-validation
            stack_method='predict_proba',  # Use probabilities
            n_jobs=-1  # Use all CPU cores
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"   Train accuracy: {train_score:.2%}")
        print(f"   Test accuracy: {test_score:.2%}")
        
        # Get predictions for more detailed metrics
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate win rate on test set
        test_wins = (y_pred == y_test).sum()
        test_total = len(y_test)
        
        # Separate by class
        positive_samples = (y_test == 1).sum()
        negative_samples = (y_test == 0).sum()
        
        print(f"   Test set distribution: {positive_samples} wins / {negative_samples} losses")
        
        # Feature importance (from gradient boosting model)
        if hasattr(self.model.named_estimators_['gradient_boosting'], 'feature_importances_'):
            importances = self.model.named_estimators_['gradient_boosting'].feature_importances_
            top_5_idx = np.argsort(importances)[-5:][::-1]
            print(f"   Top 5 feature indices: {top_5_idx.tolist()}")
        
        # Save model
        self._save_model()
        
        self.is_trained = True
        print(f"✓ ML model trained and saved!")
        
        return True
    
    def enhance_signal(self, df: pd.DataFrame, rule_signal: Dict) -> Dict:
        """
        Enhance Experta rule-based signal with ML confidence adjustment
        
        Args:
            df: Current market data with indicators
            rule_signal: Signal from Experta rules
                {
                    'signal': 'BUY'/'SELL'/'HOLD',
                    'confidence': 75.0,
                    'reasons': [...],
                    'triggered_rules': [...]
                }
        
        Returns:
            Enhanced signal with ML-adjusted confidence
        """
        # If model not trained, return original signal
        if not self.is_trained:
            rule_signal['ml_status'] = 'Not trained'
            return rule_signal
        
        # If HOLD signal, no need for ML enhancement
        if rule_signal['signal'] == 'HOLD':
            rule_signal['ml_status'] = 'Hold - no ML adjustment'
            return rule_signal
        
        try:
            # Extract features
            features = self.extract_features(
                df,
                rule_signal['signal'],
                rule_signal['confidence'],
                rule_signal.get('triggered_rules', [])
            )
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get ML prediction
            ml_confidence = self.model.predict_proba(features_scaled)[0][1]  # Probability of success
            
            # Original rule confidence (0-100)
            rule_confidence = rule_signal['confidence']
            
            # Combine: 70% rules + 30% ML
            # This keeps explainability while adding ML enhancement
            enhanced_confidence = (rule_confidence * 0.7) + (ml_confidence * 100 * 0.3)
            
            # ML bias adjustment
            ml_adjustment = enhanced_confidence - rule_confidence
            
            # Prepare enhanced signal
            enhanced_signal = rule_signal.copy()
            enhanced_signal['confidence'] = round(enhanced_confidence, 1)
            enhanced_signal['ml_confidence'] = round(ml_confidence * 100, 1)
            enhanced_signal['ml_adjustment'] = round(ml_adjustment, 1)
            enhanced_signal['ml_status'] = 'Enhanced'
            
            # Add ML insight to reasons
            if ml_adjustment > 5:
                enhanced_signal['reasons'].insert(1, f"ML boost: +{ml_adjustment:.1f}% (pattern recognition)")
            elif ml_adjustment < -5:
                enhanced_signal['reasons'].insert(1, f"ML caution: {ml_adjustment:.1f}% (risk detected)")
            
            return enhanced_signal
            
        except Exception as e:
            print(f"⚠ ML enhancement failed: {e}")
            rule_signal['ml_status'] = f'Error: {e}'
            return rule_signal
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained model"""
        if not self.is_trained or self.model is None:
            return {}
        
        # Extended feature names (matching extract_features)
        feature_names = [
            'rule_signal', 'rule_confidence', 'num_rules',
            'rsi', 'adx', 'cci', 'macd_hist', 'stochrsi',
            'atr_pct', 'bb_position', 'cmf', 'mfi',
            'trend_score', 'ema_alignment', 'supertrend', 'recent_momentum',
            # Multi-timeframe features
            'htf_trend_alignment', 'htf_momentum_strength', 'htf_rsi_divergence',
            'htf_macd_confirmation', 'htf_strong_trend',
            # Interaction features
            'rsi_volume_interaction', 'macd_adx_interaction', 'bb_volatility_interaction',
            # Momentum features
            'rsi_momentum', 'macd_momentum', 'volume_momentum'
        ]
        
        try:
            # Try to get from gradient boosting estimator
            if hasattr(self.model, 'named_estimators_'):
                if 'gradient_boosting' in self.model.named_estimators_:
                    importances = self.model.named_estimators_['gradient_boosting'].feature_importances_
                elif 'xgboost' in self.model.named_estimators_:
                    importances = self.model.named_estimators_['xgboost'].feature_importances_
                elif 'random_forest' in self.model.named_estimators_:
                    importances = self.model.named_estimators_['random_forest'].feature_importances_
                else:
                    return {}
            elif hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            else:
                return {}
            
            # Ensure we have the right number of features
            if len(importances) != len(feature_names):
                # Adjust feature names to match
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            return dict(zip(feature_names, importances))
        except Exception as e:
            print(f"⚠ Could not get feature importance: {e}")
            return {}


class HybridStrategy:
    """
    Hybrid Strategy: Experta Rules + ML Enhancement
    
    Flow:
    1. Experta generates rule-based signal (explainable)
    2. ML model adjusts confidence based on learned patterns
    3. Final decision combines both (70% rules + 30% ML)
    
    Benefits:
    - Explainability from rules
    - Pattern recognition from ML
    - Bias correction from historical learning
    - Best of both worlds!
    
    Multi-Symbol/Timeframe:
    - Automatically loads correct model for current symbol+timeframe
    - Each market has its own learned patterns
    """
    
    def __init__(self, config, experta_strategy, ml_enhancer: Optional[MLSignalEnhancer] = None):
        self.config = config
        self.experta_strategy = experta_strategy
        
        # If no ML enhancer provided, create one for this symbol+timeframe
        if ml_enhancer is None:
            symbol = getattr(config, 'trading_pair', None)
            timeframe = getattr(config, 'timeframe', None)
            
            if symbol and timeframe:
                # Create symbol-specific model
                self.ml_enhancer = MLSignalEnhancer(symbol=symbol, timeframe=timeframe)
            else:
                # Fallback to default
                self.ml_enhancer = MLSignalEnhancer()
        else:
            self.ml_enhancer = ml_enhancer
        
        self.name = "Hybrid AI Strategy (Experta + ML)"
        
    def get_name(self) -> str:
        """Return strategy name"""
        ml_status = "ML-Enhanced" if self.ml_enhancer.is_trained else "Rules-Only"
        return f"{self.name} [{ml_status}]"
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyze market using hybrid approach
        
        Steps:
        1. Get rule-based signal from Experta
        2. Enhance with ML confidence adjustment
        3. Return hybrid signal
        """
        # Step 1: Get rule-based signal (explainable)
        rule_signal = self.experta_strategy.analyze(df)
        
        # Step 2: Enhance with ML (if trained)
        if self.ml_enhancer.is_trained:
            enhanced_signal = self.ml_enhancer.enhance_signal(df, rule_signal)
            return enhanced_signal
        else:
            # No ML enhancement - return original
            rule_signal['ml_status'] = 'ML not trained - using rules only'
            return rule_signal
    
    def train_ml_model(self, historical_data: pd.DataFrame, 
                       signals: list, outcomes: list) -> bool:
        """
        Train ML model on historical trading data
        
        Args:
            historical_data: DataFrame with OHLCV and indicators
            signals: List of historical signals from Experta
            outcomes: List of outcomes (1 = profitable, 0 = loss/break-even)
        
        Returns:
            True if training successful
        """
        return self.ml_enhancer.train(historical_data, signals, outcomes)
    
    def get_model_info(self) -> Dict:
        """Get information about ML model"""
        if not self.ml_enhancer.is_trained:
            return {
                'status': 'Not trained',
                'message': 'Model will use rules only until trained'
            }
        
        feature_importance = self.ml_enhancer.get_feature_importance()
        
        # Get top 5 most important features
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'status': 'Trained',
            'model_path': self.ml_enhancer.model_path,
            'top_features': top_features
        }


def create_training_data_from_database(db_path: str = 'bot.db') -> Tuple[pd.DataFrame, list, list]:
    """
    Create training data from database history
    
    Returns:
        (historical_df, signals_list, outcomes_list)
    """
    import sqlite3
    from indicators import TechnicalIndicators
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get closed trades with P&L
    cursor.execute("""
        SELECT timestamp, symbol, side, price, pnl
        FROM trades
        WHERE pnl IS NOT NULL
        ORDER BY timestamp
    """)
    
    trades = cursor.fetchall()
    conn.close()
    
    if len(trades) < 20:
        print(f"⚠ Only {len(trades)} trades in database. Need more history to train.")
        return None, [], []
    
    print(f"✓ Found {len(trades)} historical trades")
    
    # This is simplified - in production, you'd:
    # 1. Store signal details in database
    # 2. Match signals with outcomes
    # 3. Fetch market data for each signal
    
    # For now, return empty - model will be trained manually
    return pd.DataFrame(), [], []


# ==================== EXAMPLE USAGE ====================

def example_hybrid_usage():
    """Example of how to use hybrid strategy"""
    from strategy import Strategy
    from config import Config
    
    # Create config
    config = Config()
    config.trading_pair = 'BTCUSDT'
    
    # Create Experta strategy
    experta_strategy = Strategy(config)
    
    # Create ML enhancer
    ml_enhancer = MLSignalEnhancer()
    
    # Create hybrid strategy
    hybrid = HybridStrategy(config, experta_strategy, ml_enhancer)
    
    print(f"Strategy: {hybrid.get_name()}")
    
    # If you have historical data and want to train:
    # historical_df = ...  # Your historical OHLCV with indicators
    # signals = [...]      # Historical signals from Experta
    # outcomes = [...]     # Trade outcomes (1 = profit, 0 = loss)
    # hybrid.train_ml_model(historical_df, signals, outcomes)
    
    # Use hybrid strategy
    # current_df = ...  # Current market data
    # signal = hybrid.analyze(current_df)
    # 
    # Signal will have:
    # - signal['signal']: BUY/SELL/HOLD (from rules)
    # - signal['confidence']: Enhanced confidence (rules + ML)
    # - signal['ml_adjustment']: How much ML adjusted (+/- %)
    # - signal['reasons']: Explainable reasons from rules
    # - signal['triggered_rules']: Which rules fired


if __name__ == '__main__':
    example_hybrid_usage()

