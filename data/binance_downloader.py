"""
Binance Historical Data Downloader
Downloads comprehensive OHLCV data for multiple symbols and timeframes
Stores efficiently in Parquet format
"""

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pyarrow as pa
import pyarrow.parquet as pq


class BinanceDataDownloader:
    """
    Download and store historical data from Binance
    
    Features:
    - Bulk download with rate limiting (1200 req/min)
    - Efficient Parquet storage (10x smaller, 100x faster)
    - Automatic gap filling and update
    - Resume capability for interrupted downloads
    """
    
    # Binance rate limit: 1200 requests per minute
    MAX_REQUESTS_PER_MINUTE = 1200
    REQUEST_WEIGHT_LIMIT = 2400  # Weight limit per minute
    
    # Binance max candles per request
    MAX_CANDLES_PER_REQUEST = 1500
    
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 testnet: bool = False, cache_dir: str = 'data/cached'):
        """
        Initialize downloader
        
        Args:
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
            testnet: Use testnet (limited historical data)
            cache_dir: Directory to store Parquet files
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize client
        if api_key and api_secret:
            self.client = Client(api_key, api_secret, testnet=testnet)
        else:
            # Use public client (no authentication needed for klines)
            self.client = Client("", "", testnet=testnet)
        
        # Rate limiting
        self.request_times = []
        self.total_requests = 0
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting to avoid Binance bans"""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # If we're approaching the limit, wait
        if len(self.request_times) >= self.MAX_REQUESTS_PER_MINUTE - 10:
            oldest = self.request_times[0]
            wait_time = 60 - (now - oldest) + 1
            if wait_time > 0:
                print(f"   Rate limit protection: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.request_times = []
        
        self.request_times.append(now)
        self.total_requests += 1
    
    def _get_earliest_valid_timestamp(self, symbol: str, interval: str) -> int:
        """
        Get the earliest valid timestamp for a symbol
        
        Binance futures typically have data from:
        - 2019-09-08 for major pairs (BTCUSDT, ETHUSDT)
        - Later dates for newer pairs
        """
        # Try to fetch oldest candle
        try:
            self._wait_for_rate_limit()
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=1,
                startTime=0
            )
            if klines:
                return int(klines[0][0])
        except Exception as e:
            print(f"   Warning: Could not get earliest timestamp: {e}")
        
        # Default to 3 years ago
        return int((datetime.now() - timedelta(days=1095)).timestamp() * 1000)
    
    def download_symbol(self, symbol: str, interval: str, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       force_update: bool = False) -> pd.DataFrame:
        """
        Download historical data for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
            start_date: Start date (None = earliest available)
            end_date: End date (None = now)
            force_update: Ignore cached data
        
        Returns:
            DataFrame with OHLCV data
        """
        print(f"\n{'='*70}")
        print(f"Downloading: {symbol} {interval}")
        print(f"{'='*70}")
        
        # Generate cache filename
        cache_file = os.path.join(
            self.cache_dir, 
            f"{symbol}_{interval}.parquet"
        )
        
        # Check if cached data exists
        if os.path.exists(cache_file) and not force_update:
            print(f"✓ Found cached data: {cache_file}")
            cached_df = pd.read_parquet(cache_file)
            print(f"  Cached range: {cached_df.iloc[0]['timestamp']} to {cached_df.iloc[-1]['timestamp']}")
            print(f"  Cached candles: {len(cached_df)}")
            
            # Check if we need to update
            last_cached_time = pd.to_datetime(cached_df.iloc[-1]['timestamp'])
            now = datetime.now()
            
            # If cached data is recent (within 1 hour), return it
            if (now - last_cached_time).total_seconds() < 3600:
                print(f"✓ Cache is up-to-date")
                return cached_df
            else:
                print(f"  Updating cache with recent data...")
                start_date = last_cached_time
        
        # Determine time range
        if start_date is None:
            start_timestamp = self._get_earliest_valid_timestamp(symbol, interval)
            start_date = datetime.fromtimestamp(start_timestamp / 1000)
        else:
            start_timestamp = int(start_date.timestamp() * 1000)
        
        if end_date is None:
            end_timestamp = int(datetime.now().timestamp() * 1000)
        else:
            end_timestamp = int(end_date.timestamp() * 1000)
        
        print(f"  Start: {start_date}")
        print(f"  End: {datetime.fromtimestamp(end_timestamp / 1000)}")
        
        # Download in chunks
        all_klines = []
        current_timestamp = start_timestamp
        chunk_count = 0
        
        while current_timestamp < end_timestamp:
            try:
                # Rate limiting
                self._wait_for_rate_limit()
                
                # Fetch chunk
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=current_timestamp,
                    endTime=end_timestamp,
                    limit=self.MAX_CANDLES_PER_REQUEST
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                chunk_count += 1
                
                # Update progress
                current_timestamp = int(klines[-1][0]) + 1
                progress_date = datetime.fromtimestamp(current_timestamp / 1000)
                
                if chunk_count % 10 == 0:
                    print(f"  Downloaded: {len(all_klines)} candles | Progress: {progress_date}")
                
                # If we got less than the limit, we're done
                if len(klines) < self.MAX_CANDLES_PER_REQUEST:
                    break
                    
            except BinanceAPIException as e:
                if e.code == -1121:  # Invalid symbol
                    print(f"  Error: Invalid symbol {symbol}")
                    return pd.DataFrame()
                elif e.code == -1003:  # Rate limit
                    print(f"  Rate limit hit, waiting 60s...")
                    time.sleep(60)
                    continue
                else:
                    print(f"  API Error: {e}")
                    break
            except Exception as e:
                print(f"  Error: {e}")
                break
        
        if not all_klines:
            print(f"  No data downloaded")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Keep only essential columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Merge with cached data if exists
        if os.path.exists(cache_file) and not force_update:
            cached_df = pd.read_parquet(cache_file)
            df = pd.concat([cached_df, df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        # Save to Parquet
        df.to_parquet(cache_file, compression='snappy', index=False)
        
        print(f"\n✓ Download complete!")
        print(f"  Total candles: {len(df)}")
        print(f"  Range: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
        print(f"  Saved to: {cache_file}")
        print(f"  File size: {os.path.getsize(cache_file) / 1024 / 1024:.2f} MB")
        print(f"  Total requests: {self.total_requests}")
        
        return df
    
    def download_multiple(self, symbols: List[str], intervals: List[str],
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Download multiple symbols and timeframes
        
        Args:
            symbols: List of symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            intervals: List of timeframes (e.g., ['5m', '15m', '1h'])
            start_date: Start date (None = earliest available)
            end_date: End date (None = now)
        
        Returns:
            Nested dict: {symbol: {interval: dataframe}}
        """
        print(f"\n{'═'*70}")
        print(f"BULK DOWNLOAD")
        print(f"{'═'*70}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Intervals: {', '.join(intervals)}")
        print(f"Total combinations: {len(symbols) * len(intervals)}")
        print()
        
        results = {}
        total_combinations = len(symbols) * len(intervals)
        current = 0
        
        start_time = time.time()
        
        for symbol in symbols:
            results[symbol] = {}
            for interval in intervals:
                current += 1
                print(f"\n[{current}/{total_combinations}] {symbol} {interval}")
                
                try:
                    df = self.download_symbol(symbol, interval, start_date, end_date)
                    results[symbol][interval] = df
                    
                    # Show progress
                    elapsed = time.time() - start_time
                    avg_time = elapsed / current
                    remaining = (total_combinations - current) * avg_time
                    print(f"  ETA: {remaining / 60:.1f} minutes")
                    
                except Exception as e:
                    print(f"  Failed: {e}")
                    results[symbol][interval] = pd.DataFrame()
        
        print(f"\n{'═'*70}")
        print(f"BULK DOWNLOAD COMPLETE")
        print(f"{'═'*70}")
        print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
        print(f"Total requests: {self.total_requests}")
        print(f"Success rate: {sum(1 for s in results.values() for df in s.values() if len(df) > 0)}/{total_combinations}")
        
        return results
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available USDT futures symbols"""
        try:
            exchange_info = self.client.futures_exchange_info()
            symbols = [
                s['symbol'] for s in exchange_info['symbols']
                if s['symbol'].endswith('USDT') and s['status'] == 'TRADING'
            ]
            return sorted(symbols)
        except Exception as e:
            print(f"Error getting symbols: {e}")
            return []
    
    def get_top_volume_symbols(self, limit: int = 20) -> List[str]:
        """Get top symbols by 24h volume"""
        try:
            tickers = self.client.futures_ticker()
            
            # Filter USDT pairs and sort by volume
            usdt_tickers = [
                t for t in tickers 
                if t['symbol'].endswith('USDT')
            ]
            
            sorted_tickers = sorted(
                usdt_tickers,
                key=lambda x: float(x.get('quoteVolume', 0)),
                reverse=True
            )
            
            return [t['symbol'] for t in sorted_tickers[:limit]]
        except Exception as e:
            print(f"Error getting top symbols: {e}")
            return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Fallback

