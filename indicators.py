"""
Professional Technical Indicators powered by pandas_ta
Full-featured implementation with advanced indicators
Optimized for PC environments
"""

import pandas as pd
import numpy as np
import pandas_ta as ta


class TechnicalIndicators:
    """
    Calculate technical indicators using pandas_ta
    Now with full power - ADX, ATR, Stochastic, Supertrend, and more!
    """
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators to dataframe using pandas_ta
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        
        Returns:
            DataFrame with all indicators added
        """
        result = df.copy()
        
        # Ensure we have the right column names for pandas_ta
        result.columns = result.columns.str.lower()
        
        # Set up DatetimeIndex for VWAP (if timestamp column exists)
        has_timestamp_column = False
        if 'timestamp' in result.columns:
            has_timestamp_column = True
            # Convert timestamp to datetime and set as index
            result['timestamp'] = pd.to_datetime(result['timestamp'])
            result = result.set_index('timestamp').sort_index()
        
        # === TREND INDICATORS ===
        
        # EMAs (Multiple timeframes)
        result['ema_9'] = ta.ema(result['close'], length=9)
        result['ema_21'] = ta.ema(result['close'], length=21)
        result['ema_50'] = ta.ema(result['close'], length=50)
        result['ema_200'] = ta.ema(result['close'], length=200)
        
        # SMAs (Alternative moving averages)
        result['sma_20'] = ta.sma(result['close'], length=20)
        result['sma_50'] = ta.sma(result['close'], length=50)
        
        # MACD (Moving Average Convergence Divergence)
        macd_result = ta.macd(result['close'], fast=12, slow=26, signal=9)
        if macd_result is not None and not macd_result.empty:
            macd_cols = macd_result.columns.tolist()
            for col in macd_cols:
                if 'MACD_' in col and 'h' not in col and 's' not in col.lower():
                    result['macd'] = macd_result[col]
                elif 'MACDs' in col or 'signal' in col.lower():
                    result['macd_signal'] = macd_result[col]
                elif 'MACDh' in col or 'hist' in col.lower():
                    result['macd_hist'] = macd_result[col]
        
        # ADX (Average Directional Index) - Trend Strength
        adx_result = ta.adx(result['high'], result['low'], result['close'], length=14)
        if adx_result is not None and not adx_result.empty:
            adx_cols = adx_result.columns.tolist()
            for col in adx_cols:
                if 'ADX' in col:
                    result['adx'] = adx_result[col]
                elif 'DMP' in col:
                    result['dmp'] = adx_result[col]
                elif 'DMN' in col:
                    result['dmn'] = adx_result[col]
        
        # Supertrend (Advanced trend following)
        supertrend_result = ta.supertrend(result['high'], result['low'], result['close'], 
                                          length=10, multiplier=3.0)
        if supertrend_result is not None and not supertrend_result.empty:
            st_cols = supertrend_result.columns.tolist()
            for col in st_cols:
                if 'SUPERTd' in col:
                    result['supertrend_direction'] = supertrend_result[col]
                elif 'SUPERT' in col:
                    result['supertrend'] = supertrend_result[col]
        
        # === MOMENTUM INDICATORS ===
        
        # RSI (Relative Strength Index)
        result['rsi'] = ta.rsi(result['close'], length=14)
        
        # Stochastic RSI (More sensitive)
        stochrsi_result = ta.stochrsi(result['close'], length=14, rsi_length=14, k=3, d=3)
        if stochrsi_result is not None and not stochrsi_result.empty:
            srsi_cols = stochrsi_result.columns.tolist()
            for col in srsi_cols:
                if 'STOCHRSIk' in col or ('k' in col.lower() and 'stochrsi' in col.lower()):
                    result['stochrsi_k'] = stochrsi_result[col]
                elif 'STOCHRSId' in col or ('d' in col.lower() and 'stochrsi' in col.lower()):
                    result['stochrsi_d'] = stochrsi_result[col]
        
        # Stochastic Oscillator (Classic)
        stoch_result = ta.stoch(result['high'], result['low'], result['close'], 
                               k=14, d=3, smooth_k=3)
        if stoch_result is not None and not stoch_result.empty:
            stoch_cols = stoch_result.columns.tolist()
            for col in stoch_cols:
                if 'STOCHk' in col:
                    result['stoch_k'] = stoch_result[col]
                elif 'STOCHd' in col:
                    result['stoch_d'] = stoch_result[col]
        
        # CCI (Commodity Channel Index)
        result['cci'] = ta.cci(result['high'], result['low'], result['close'], length=20)
        
        # Williams %R
        result['willr'] = ta.willr(result['high'], result['low'], result['close'], length=14)
        
        # === VOLATILITY INDICATORS ===
        
        # Bollinger Bands
        bb_result = ta.bbands(result['close'], length=20, std=2.0)
        if bb_result is not None and not bb_result.empty:
            # pandas_ta column names can vary, find them dynamically
            bb_cols = bb_result.columns.tolist()
            for col in bb_cols:
                if 'BBL' in col:
                    result['bb_lower'] = bb_result[col]
                elif 'BBM' in col:
                    result['bb_middle'] = bb_result[col]
                elif 'BBU' in col:
                    result['bb_upper'] = bb_result[col]
                elif 'BBB' in col:
                    result['bb_bandwidth'] = bb_result[col]
                elif 'BBP' in col:
                    result['bb_percent'] = bb_result[col]
        
        # ATR (Average True Range) - Volatility measure
        result['atr'] = ta.atr(result['high'], result['low'], result['close'], length=14)
        
        # Keltner Channels (Alternative to Bollinger)
        kc_result = ta.kc(result['high'], result['low'], result['close'], 
                         length=20, scalar=2.0)
        if kc_result is not None and not kc_result.empty:
            kc_cols = kc_result.columns.tolist()
            for col in kc_cols:
                if 'KCL' in col:
                    result['kc_lower'] = kc_result[col]
                elif 'KCB' in col:
                    result['kc_basis'] = kc_result[col]
                elif 'KCU' in col:
                    result['kc_upper'] = kc_result[col]
        
        # Donchian Channels (Breakout indicator)
        dc_result = ta.donchian(result['high'], result['low'], lower_length=20, upper_length=20)
        if dc_result is not None and not dc_result.empty:
            dc_cols = dc_result.columns.tolist()
            for col in dc_cols:
                if 'DCL' in col:
                    result['dc_lower'] = dc_result[col]
                elif 'DCM' in col:
                    result['dc_middle'] = dc_result[col]
                elif 'DCU' in col:
                    result['dc_upper'] = dc_result[col]
        
        # === VOLUME INDICATORS ===
        
        # OBV (On Balance Volume)
        result['obv'] = ta.obv(result['close'], result['volume'])
        
        # Volume SMA (Volume trend)
        result['volume_sma'] = ta.sma(result['volume'], length=20)
        
        # CMF (Chaikin Money Flow)
        result['cmf'] = ta.cmf(result['high'], result['low'], result['close'], 
                              result['volume'], length=20)
        
        # MFI (Money Flow Index) - Volume-weighted RSI
        result['mfi'] = ta.mfi(result['high'], result['low'], result['close'], 
                              result['volume'], length=14)
        
        # VWAP (Volume Weighted Average Price) - Intraday benchmark
        # VWAP requires DatetimeIndex (already set above if timestamp exists)
        try:
            result['vwap'] = ta.vwap(result['high'], result['low'], result['close'], 
                                    result['volume'])
        except Exception:
            # If VWAP fails (no proper index), set to NaN
            result['vwap'] = pd.Series(float('nan'), index=result.index)
        
        # === PATTERN & SUPPORT/RESISTANCE ===
        
        # Ichimoku Cloud (Japanese indicator suite)
        try:
            ichimoku_result = ta.ichimoku(result['high'], result['low'], result['close'])
            if ichimoku_result is not None and len(ichimoku_result) > 0:
                ich = ichimoku_result[0] if isinstance(ichimoku_result, tuple) else ichimoku_result
                if not ich.empty:
                    ich_cols = ich.columns.tolist()
                    for col in ich_cols:
                        if 'ISA' in col:
                            result['ichimoku_a'] = ich[col]
                        elif 'ISB' in col:
                            result['ichimoku_b'] = ich[col]
                        elif 'ITS' in col:
                            result['ichimoku_base'] = ich[col]
                        elif 'IKS' in col:
                            result['ichimoku_conv'] = ich[col]
        except:
            pass  # Ichimoku can fail on short data
        
        # Pivot Points (Support/Resistance levels)
        try:
            pivot_result = ta.pivot(result['high'], result['low'], result['close'])
            if pivot_result is not None and not pivot_result.empty:
                piv_cols = pivot_result.columns.tolist()
                for col in piv_cols:
                    if col == 'PP' or 'pivot' in col.lower():
                        result['pivot'] = pivot_result[col]
                        break
        except:
            pass  # Pivot can fail on short data
            
        # === CUSTOM CALCULATED INDICATORS ===
        
        # EMA Ribbon Strength (trend strength based on EMA alignment)
        result['ema_ribbon_strength'] = TechnicalIndicators._calculate_ema_ribbon_strength(result)
        
        # Volume Pressure (buying vs selling pressure)
        result['volume_pressure'] = TechnicalIndicators._calculate_volume_pressure(result)
        
        # Trend Score (composite trend indicator)
        result['trend_score'] = TechnicalIndicators._calculate_trend_score(result)
        
        # Reset index if we set it for VWAP (restore timestamp column)
        if has_timestamp_column and result.index.name == 'timestamp':
            result = result.reset_index()
        
        return result
    
    @staticmethod
    def _calculate_ema_ribbon_strength(df: pd.DataFrame) -> pd.Series:
        """
        Calculate EMA ribbon strength
        Positive = bullish alignment, Negative = bearish alignment
        """
        if 'ema_9' not in df.columns or 'ema_21' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Check if EMAs are properly aligned
        ema_9 = df['ema_9']
        ema_21 = df['ema_21']
        ema_50 = df.get('ema_50', ema_21)
        
        # Calculate separation percentages
        sep_9_21 = ((ema_9 - ema_21) / ema_21) * 100
        sep_21_50 = ((ema_21 - ema_50) / ema_50) * 100
        
        # Combine (positive = bullish, negative = bearish)
        strength = (sep_9_21 + sep_21_50) / 2
        
        return strength
    
    @staticmethod
    def _calculate_volume_pressure(df: pd.DataFrame) -> pd.Series:
        """
        Calculate buying/selling pressure based on price movement and volume
        Positive = buying pressure, Negative = selling pressure
        """
        if 'volume' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Price change
        price_change = df['close'].pct_change()
        
        # Volume relative to average
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'] / volume_ma
        
        # Combine: strong volume + price increase = buying pressure
        pressure = price_change * volume_ratio * 100
        
        return pressure
    
    @staticmethod
    def _calculate_trend_score(df: pd.DataFrame) -> pd.Series:
        """
        Composite trend score from multiple indicators
        Range: -100 (strong downtrend) to +100 (strong uptrend)
        """
        score = pd.Series(0, index=df.index)
        weights = 0
        
        # 1. EMA alignment (40% weight)
        if all(col in df.columns for col in ['ema_9', 'ema_21', 'ema_50']):
            # Check if we have valid data (not all NaN/None)
            if df['ema_9'].notna().sum() > 0 and df['ema_21'].notna().sum() > 0 and df['ema_50'].notna().sum() > 0:
                ema_bullish = (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])
                ema_bearish = (df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50'])
                score += ema_bullish * 40 - ema_bearish * 40
                weights += 40
        
        # 2. MACD (30% weight)
        if 'macd_hist' in df.columns and df['macd_hist'].notna().sum() > 0:
            macd_score = df['macd_hist'].clip(-1, 1) * 30
            score += macd_score
            weights += 30
        
        # 3. ADX trend strength (30% weight)
        if all(col in df.columns for col in ['adx', 'dmp', 'dmn']):
            if df['adx'].notna().sum() > 0 and df['dmp'].notna().sum() > 0 and df['dmn'].notna().sum() > 0:
                # Strong trend + direction
                trend_strength = df['adx'].clip(0, 50) / 50  # Normalize to 0-1
                direction = (df['dmp'] > df['dmn']) * 2 - 1  # +1 for bullish, -1 for bearish
                score += trend_strength * direction * 30
                weights += 30
        
        # Normalize to -100 to +100 range
        if weights > 0:
            score = (score / weights) * 100
        
        return score
    
    @staticmethod
    def add_higher_timeframe_indicators(primary_df: pd.DataFrame,
                                       higher_df: pd.DataFrame,
                                       htf_suffix: str = 'htf') -> pd.DataFrame:
        """
        Add higher timeframe indicators to primary timeframe
        
        This is crucial for multi-timeframe analysis:
        - Don't buy on 5m if 1h is in downtrend
        - Confirm signals with higher timeframe momentum
        - Reduce false signals by 30-40%
        
        Args:
            primary_df: Primary timeframe data (e.g., 5m)
            higher_df: Higher timeframe data (e.g., 1h)
            htf_suffix: Suffix for HTF columns (e.g., 'htf_1h')
        
        Returns:
            Primary dataframe with HTF indicators added
        """
        result = primary_df.copy()
        
        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(result['timestamp']):
            result['timestamp'] = pd.to_datetime(result['timestamp'])
        if not pd.api.types.is_datetime64_any_dtype(higher_df['timestamp']):
            higher_df = higher_df.copy()
            higher_df['timestamp'] = pd.to_datetime(higher_df['timestamp'])
        
        # Select key HTF indicators to merge
        htf_indicators = [
            'rsi', 'macd', 'macd_hist', 'adx', 'trend_score',
            'ema_9', 'ema_21', 'ema_50', 'supertrend_direction',
            'dmp', 'dmn', 'stochrsi_k', 'bb_position'
        ]
        
        # Create HTF dataframe with selected indicators
        htf_cols = ['timestamp'] + [col for col in htf_indicators if col in higher_df.columns]
        htf_data = higher_df[htf_cols].copy()
        
        # Rename columns with suffix
        rename_dict = {col: f"{col}_{htf_suffix}" for col in htf_data.columns if col != 'timestamp'}
        htf_data.rename(columns=rename_dict, inplace=True)
        
        # Merge using backward fill (each HTF candle applies to multiple primary candles)
        result = pd.merge_asof(
            result.sort_values('timestamp'),
            htf_data.sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        
        # Calculate HTF-specific features
        
        # 1. HTF Trend Alignment Score (-2 to +2)
        # +2 = strong uptrend, -2 = strong downtrend, 0 = neutral
        htf_trend_score = 0
        
        if f'trend_score_{htf_suffix}' in result.columns:
            htf_trend = result[f'trend_score_{htf_suffix}']
            htf_trend_score = (htf_trend / 50).clip(-2, 2)  # Normalize
        elif all(f'{col}_{htf_suffix}' in result.columns for col in ['ema_9', 'ema_21', 'ema_50']):
            # Calculate from EMAs if trend_score not available
            ema9_htf = result[f'ema_9_{htf_suffix}']
            ema21_htf = result[f'ema_21_{htf_suffix}']
            ema50_htf = result[f'ema_50_{htf_suffix}']
            
            bullish = (ema9_htf > ema21_htf) & (ema21_htf > ema50_htf)
            bearish = (ema9_htf < ema21_htf) & (ema21_htf < ema50_htf)
            
            htf_trend_score = bullish.astype(int) * 2 - bearish.astype(int) * 2
        
        result[f'htf_trend_alignment'] = htf_trend_score
        
        # 2. HTF Momentum Confirmation (0 to 1)
        # High value = HTF momentum confirms primary TF direction
        if f'rsi_{htf_suffix}' in result.columns:
            htf_rsi = result[f'rsi_{htf_suffix}']
            # RSI > 50 = bullish, < 50 = bearish
            result[f'htf_momentum_strength'] = ((htf_rsi - 50) / 50).clip(-1, 1)
        
        # 3. HTF-Primary RSI Divergence
        # Detect when timeframes disagree (potential reversal signal)
        if 'rsi' in result.columns and f'rsi_{htf_suffix}' in result.columns:
            rsi_diff = result['rsi'] - result[f'rsi_{htf_suffix}']
            result[f'htf_rsi_divergence'] = rsi_diff
        
        # 4. HTF MACD Confirmation
        if f'macd_hist_{htf_suffix}' in result.columns:
            macd_hist_htf = result[f'macd_hist_{htf_suffix}']
            result[f'htf_macd_bullish'] = (macd_hist_htf > 0).astype(int)
            result[f'htf_macd_bearish'] = (macd_hist_htf < 0).astype(int)
        
        # 5. HTF ADX (Trend Strength)
        if f'adx_{htf_suffix}' in result.columns:
            htf_adx = result[f'adx_{htf_suffix}']
            # Strong trend if ADX > 25
            result[f'htf_strong_trend'] = (htf_adx > 25).astype(int)
        
        return result
    
    @staticmethod
    def get_indicator_summary(df: pd.DataFrame) -> dict:
        """
        Get a summary of key indicator signals
        Useful for quick market analysis
        """
        if len(df) == 0:
            return {}
        
        latest = df.iloc[-1]
        summary = {}
        
        # Trend
        if 'trend_score' in latest:
            trend_score = latest['trend_score']
            if trend_score > 30:
                summary['trend'] = 'STRONG UPTREND'
            elif trend_score > 10:
                summary['trend'] = 'UPTREND'
            elif trend_score < -30:
                summary['trend'] = 'STRONG DOWNTREND'
            elif trend_score < -10:
                summary['trend'] = 'DOWNTREND'
            else:
                summary['trend'] = 'RANGING'
        
        # Momentum
        if 'rsi' in latest:
            rsi = latest['rsi']
            if rsi > 70:
                summary['momentum'] = 'OVERBOUGHT'
            elif rsi < 30:
                summary['momentum'] = 'OVERSOLD'
            else:
                summary['momentum'] = 'NEUTRAL'
        
        # Volatility
        if 'atr' in latest and 'close' in latest:
            atr_pct = (latest['atr'] / latest['close']) * 100
            if atr_pct > 3:
                summary['volatility'] = 'HIGH'
            elif atr_pct < 1:
                summary['volatility'] = 'LOW'
            else:
                summary['volatility'] = 'NORMAL'
        
        # Volume
        if 'volume_pressure' in latest:
            vol_pressure = latest['volume_pressure']
            if vol_pressure > 2:
                summary['volume'] = 'STRONG BUYING'
            elif vol_pressure < -2:
                summary['volume'] = 'STRONG SELLING'
            else:
                summary['volume'] = 'NEUTRAL'
        
        # Higher Timeframe (if available)
        if 'htf_trend_alignment' in latest:
            htf_trend = latest['htf_trend_alignment']
            if htf_trend > 1:
                summary['htf_trend'] = 'STRONG BULLISH'
            elif htf_trend > 0:
                summary['htf_trend'] = 'BULLISH'
            elif htf_trend < -1:
                summary['htf_trend'] = 'STRONG BEARISH'
            elif htf_trend < 0:
                summary['htf_trend'] = 'BEARISH'
            else:
                summary['htf_trend'] = 'NEUTRAL'
        
        return summary

