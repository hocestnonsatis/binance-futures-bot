"""
Machine Learning Model for Signal Enhancement
Hybrid AI: Experta Rules + ML Confidence Adjustment
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
    
    def __init__(self, symbol: str = None, timeframe: str = None, model_dir: str = 'models'):
        """
        Initialize ML Signal Enhancer
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Candle timeframe (e.g., '5m', '15m', '1h')
            model_dir: Directory to store models
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_dir = model_dir
        
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
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model if exists"""
        if os.path.exists(self.model_path):
            try:
                saved_data = joblib.load(self.model_path)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.is_trained = True
                
                # Show which model loaded
                if self.symbol and self.timeframe:
                    print(f"✓ ML model loaded: {self.symbol} {self.timeframe}")
                else:
                    model_name = os.path.basename(self.model_path)
                    print(f"✓ ML model loaded: {model_name}")
            except Exception as e:
                print(f"⚠ Could not load model: {e}")
                self.is_trained = False
    
    def get_model_info(self) -> Dict:
        """Get information about this model instance"""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'model_path': self.model_path,
            'is_trained': self.is_trained,
            'exists': os.path.exists(self.model_path)
        }
    
    def auto_train(self, binance_client, limit: int = 1000) -> bool:
        """
        Automatically train model using Binance data
        
        Args:
            binance_client: BinanceFutures instance
            limit: Number of candles to fetch (default 1000)
        
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
            import pandas as pd
            import numpy as np
            
            # Fetch historical data
            print(f"📥 Fetching {limit} candles of {self.symbol} {self.timeframe}...")
            df = binance_client.get_klines(self.symbol, self.timeframe, limit=limit)
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
            
            # Generate training samples
            print("🔄 Generating training signals...")
            signals = []
            outcomes = []
            
            for i in range(150, len(df) - 10, 2):  # Every 2nd candle, leave 10 for outcome check
                current = df.iloc[i]
                df_slice = df.iloc[:i+1]
                
                # Get indicators
                rsi = current.get('rsi')
                macd_hist = current.get('macd_hist')
                adx = current.get('adx')
                
                if pd.isna(rsi) or pd.isna(macd_hist):
                    continue
                
                # Create synthetic signals based on strong patterns
                signal_dict = None
                
                # Bullish patterns
                if rsi < 40 and macd_hist > 0:
                    signal_dict = {
                        'signal': 'BUY',
                        'confidence': 55.0 + np.random.rand() * 15,
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
                future_prices = df.iloc[i+1:i+11]['close'].values
                entry_price = current['close']
                
                if signal_dict['signal'] == 'BUY':
                    # Check if price went up
                    max_price = future_prices.max()
                    outcome = 1 if (max_price - entry_price) / entry_price > 0.01 else 0  # 1% profit
                else:  # SELL
                    # Check if price went down
                    min_price = future_prices.min()
                    outcome = 1 if (entry_price - min_price) / entry_price > 0.01 else 0
                
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
        """Save trained model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, self.model_path)
        print(f"✓ Model saved to {self.model_path}")
    
    def extract_features(self, df: pd.DataFrame, rule_signal: str, 
                        rule_confidence: float, triggered_rules: list) -> np.ndarray:
        """
        Extract features for ML model
        
        Combines:
        - Technical indicator values
        - Rule-based signal information
        - Market regime indicators
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
        features.append(latest.get('rsi', 50) / 100.0)  # Normalize
        features.append(latest.get('adx', 20) / 100.0)
        features.append(latest.get('cci', 0) / 200.0)  # Normalize CCI
        
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
        
        # Train Gradient Boosting model (better for trading)
        print("   Training Gradient Boosting model...")
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"   Train accuracy: {train_score:.2%}")
        print(f"   Test accuracy: {test_score:.2%}")
        
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
        
        feature_names = [
            'rule_signal', 'rule_confidence', 'num_rules',
            'rsi', 'adx', 'cci', 'macd_hist', 'stochrsi',
            'atr_pct', 'bb_position', 'cmf', 'mfi',
            'trend_score', 'ema_alignment', 'supertrend',
            'recent_momentum'
        ]
        
        importances = self.model.feature_importances_
        
        return dict(zip(feature_names, importances))


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

