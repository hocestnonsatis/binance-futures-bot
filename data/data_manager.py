"""
Data Manager - Centralized data access and multi-timeframe alignment
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class DataManager:
    """
    Centralized data access layer
    
    Features:
    - Load cached Parquet data efficiently
    - Align multiple timeframes
    - Automatic data validation
    - Memory caching for frequently accessed data
    """
    
    def __init__(self, cache_dir: str = 'data/cached'):
        """
        Initialize data manager
        
        Args:
            cache_dir: Directory containing Parquet files
        """
        self.cache_dir = cache_dir
        self.memory_cache = {}  # In-memory cache
        self.cache_hits = 0
        self.cache_misses = 0
    
    def load_data(self, symbol: str, interval: str, 
                  use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Load data for symbol and interval
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '5m', '1h')
            use_cache: Use memory cache
        
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        cache_key = f"{symbol}_{interval}"
        
        # Check memory cache
        if use_cache and cache_key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[cache_key].copy()
        
        self.cache_misses += 1
        
        # Load from Parquet
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            df = pd.read_parquet(cache_file)
            
            # Validate data
            if not self._validate_data(df):
                print(f"Warning: Invalid data in {cache_file}")
                return None
            
            # Store in memory cache
            if use_cache:
                self.memory_cache[cache_key] = df.copy()
            
            return df
        except Exception as e:
            print(f"Error loading {cache_file}: {e}")
            return None
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate dataframe structure"""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        if df is None or len(df) == 0:
            return False
        
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check for NaN values
        if df[required_columns].isna().any().any():
            return False
        
        # Check OHLC integrity (high >= low, etc.)
        if not ((df['high'] >= df['low']).all() and 
                (df['high'] >= df['open']).all() and 
                (df['high'] >= df['close']).all()):
            return False
        
        return True
    
    def load_multi_timeframe(self, symbol: str, intervals: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load multiple timeframes for a symbol
        
        Args:
            symbol: Trading pair
            intervals: List of timeframes (e.g., ['5m', '15m', '1h'])
        
        Returns:
            Dict mapping interval to DataFrame
        """
        result = {}
        
        for interval in intervals:
            df = self.load_data(symbol, interval)
            if df is not None:
                result[interval] = df
        
        return result
    
    def align_timeframes(self, primary_df: pd.DataFrame, 
                        primary_interval: str,
                        higher_df: pd.DataFrame,
                        higher_interval: str) -> pd.DataFrame:
        """
        Align higher timeframe data to primary timeframe
        
        This merges higher timeframe indicators into the primary timeframe
        using forward-fill (each higher TF candle applies to multiple lower TF candles)
        
        Args:
            primary_df: Primary timeframe data (e.g., 5m)
            primary_interval: Primary interval
            higher_df: Higher timeframe data (e.g., 1h)
            higher_interval: Higher interval
        
        Returns:
            Primary dataframe with higher timeframe columns added
        """
        # Create copy
        result = primary_df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(result['timestamp']):
            result['timestamp'] = pd.to_datetime(result['timestamp'])
        if not pd.api.types.is_datetime64_any_dtype(higher_df['timestamp']):
            higher_df = higher_df.copy()
            higher_df['timestamp'] = pd.to_datetime(higher_df['timestamp'])
        
        # Rename higher TF columns to include suffix
        htf_suffix = f"_htf_{higher_interval}"
        higher_renamed = higher_df.copy()
        
        for col in higher_renamed.columns:
            if col != 'timestamp':
                higher_renamed.rename(columns={col: f"{col}{htf_suffix}"}, inplace=True)
        
        # Merge using pandas merge_asof (forward fill)
        result = pd.merge_asof(
            result.sort_values('timestamp'),
            higher_renamed.sort_values('timestamp'),
            on='timestamp',
            direction='backward'  # Use the most recent higher TF candle
        )
        
        return result
    
    def resample_to_higher_timeframe(self, df: pd.DataFrame, 
                                     from_interval: str, 
                                     to_interval: str) -> pd.DataFrame:
        """
        Resample lower timeframe to higher timeframe
        
        Args:
            df: DataFrame with OHLCV data
            from_interval: Source interval (e.g., '5m')
            to_interval: Target interval (e.g., '1h')
        
        Returns:
            Resampled DataFrame
        """
        # Convert intervals to pandas frequency
        freq_map = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
        }
        
        if to_interval not in freq_map:
            raise ValueError(f"Unsupported interval: {to_interval}")
        
        target_freq = freq_map[to_interval]
        
        # Ensure timestamp is datetime and set as index
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df = df.set_index('timestamp')
        
        # Resample OHLCV
        resampled = df.resample(target_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove NaN rows (incomplete candles)
        resampled = resampled.dropna()
        
        # Reset index
        resampled = resampled.reset_index()
        
        return resampled
    
    def get_recent_data(self, symbol: str, interval: str, 
                       limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Get most recent N candles
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            limit: Number of recent candles
        
        Returns:
            DataFrame with recent data
        """
        df = self.load_data(symbol, interval)
        
        if df is None:
            return None
        
        return df.tail(limit).reset_index(drop=True)
    
    def get_data_range(self, symbol: str, interval: str,
                      start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """
        Get data for specific date range
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            start: Start datetime
            end: End datetime
        
        Returns:
            DataFrame filtered to date range
        """
        df = self.load_data(symbol, interval)
        
        if df is None:
            return None
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        return df[mask].reset_index(drop=True)
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_accesses * 100) if total_accesses > 0 else 0
        
        return {
            'cached_items': len(self.memory_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }
    
    def clear_cache(self):
        """Clear memory cache"""
        self.memory_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def list_available_data(self) -> List[Tuple[str, str, int, str]]:
        """
        List all available data files
        
        Returns:
            List of tuples: (symbol, interval, num_candles, file_size)
        """
        result = []
        
        if not os.path.exists(self.cache_dir):
            return result
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.parquet'):
                # Parse filename
                parts = filename.replace('.parquet', '').split('_')
                if len(parts) >= 2:
                    symbol = '_'.join(parts[:-1])
                    interval = parts[-1]
                    
                    filepath = os.path.join(self.cache_dir, filename)
                    file_size_mb = os.path.getsize(filepath) / 1024 / 1024
                    
                    # Load to get candle count
                    try:
                        df = pd.read_parquet(filepath)
                        num_candles = len(df)
                        result.append((symbol, interval, num_candles, f"{file_size_mb:.2f} MB"))
                    except:
                        pass
        
        return sorted(result)

