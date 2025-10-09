"""
Data management package for historical market data

This package provides:
- Binance data downloader (binance_downloader.py)
- Data manager for caching and access (data_manager.py)
- Efficient Parquet storage for fast loading
"""

from .binance_downloader import BinanceDataDownloader
from .data_manager import DataManager

__all__ = ['BinanceDataDownloader', 'DataManager']

