"""
Minimal Database for Trade History and Logging
Simple SQLite with just trades and logs tables
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import contextmanager


class Database:
    """Lightweight database manager for trades and logs"""
    
    def __init__(self, db_path: str = 'bot.db'):
        self.db_path = db_path
        self._init_tables()
    
    @contextmanager
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_tables(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    strategy TEXT,
                    order_id TEXT
                )
            ''')
            
            # Logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL
                )
            ''')
            
            # Positions table (for open positions)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    leverage INTEGER NOT NULL,
                    entry_time TEXT NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT DEFAULT 'open'
                )
            ''')
    
    # === TRADES ===
    
    def add_trade(self, symbol: str, side: str, price: float, quantity: float, 
                  strategy: str = '', pnl: float = 0, order_id: str = '') -> int:
        """Record a trade"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (timestamp, symbol, side, price, quantity, pnl, strategy, order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), symbol, side, price, quantity, pnl, strategy, order_id))
            return cursor.lastrowid
    
    def get_trades(self, limit: int = 100, symbol: str = None) -> List[Dict]:
        """Get recent trades"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute('''
                    SELECT * FROM trades 
                    WHERE symbol = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (symbol, limit))
            else:
                cursor.execute('''
                    SELECT * FROM trades 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_total_pnl(self, symbol: str = None) -> float:
        """Get total P&L"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute('SELECT SUM(pnl) FROM trades WHERE symbol = ?', (symbol,))
            else:
                cursor.execute('SELECT SUM(pnl) FROM trades')
            
            result = cursor.fetchone()[0]
            return result if result else 0.0
    
    def get_daily_pnl(self) -> float:
        """Get today's P&L"""
        today = datetime.now().date().isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT SUM(pnl) FROM trades 
                WHERE DATE(timestamp) = ?
            ''', (today,))
            result = cursor.fetchone()[0]
            return result if result else 0.0
    
    # === POSITIONS ===
    
    def add_position(self, symbol: str, side: str, entry_price: float, 
                     quantity: float, leverage: int, stop_loss: float = None, 
                     take_profit: float = None) -> int:
        """Record an open position"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO positions 
                (symbol, side, entry_price, quantity, leverage, entry_time, stop_loss, take_profit, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
            ''', (symbol, side, entry_price, quantity, leverage, datetime.now().isoformat(), 
                  stop_loss, take_profit))
            return cursor.lastrowid
    
    def get_open_position(self, symbol: str) -> Optional[Dict]:
        """Get open position for symbol"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM positions 
                WHERE symbol = ? AND status = 'open'
                ORDER BY entry_time DESC
                LIMIT 1
            ''', (symbol,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def close_position(self, position_id: int, pnl: float):
        """Close a position"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE positions 
                SET status = 'closed'
                WHERE id = ?
            ''', (position_id,))
    
    # === LOGS ===
    
    def log(self, level: str, message: str):
        """Add a log entry"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO logs (timestamp, level, message)
                VALUES (?, ?, ?)
            ''', (datetime.now().isoformat(), level, message))
    
    def info(self, message: str):
        """Log info message"""
        self.log('INFO', message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.log('WARNING', message)
    
    def error(self, message: str):
        """Log error message"""
        self.log('ERROR', message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.log('DEBUG', message)
    
    def get_logs(self, limit: int = 50, level: str = None) -> List[Dict]:
        """Get recent logs"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if level:
                cursor.execute('''
                    SELECT * FROM logs 
                    WHERE level = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (level, limit))
            else:
                cursor.execute('''
                    SELECT * FROM logs 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]

