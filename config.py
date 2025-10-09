"""
Configuration Management for Futures Trading Bot
Simple, clean configuration with CLI input and .env support
"""

import os
import json
from dotenv import load_dotenv
from typing import Dict, Optional
from datetime import datetime

# ANSI Colors for terminal
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


class Config:
    """Trading bot configuration"""
    
    CONFIG_CACHE_FILE = 'config_cache.json'
    
    def __init__(self):
        load_dotenv()
        
        # API Configuration (from .env)
        self.api_key = os.getenv('BINANCE_API_KEY', '')
        self.api_secret = os.getenv('BINANCE_API_SECRET', '')
        self.testnet = os.getenv('TESTNET', 'true').lower() == 'true'
        
        # Trading Configuration (from CLI)
        self.trading_pair = ''
        self.timeframe = '5m'
        self.leverage = 20  # Increased from 10 for higher returns
        self.margin_type = 'ISOLATED'
        
        # Multi-Timeframe Configuration
        self.use_multi_timeframe = True  # Enable multi-timeframe analysis
        self.higher_timeframe = None  # Auto-calculated based on primary timeframe
        self.htf_multiplier = 12  # Higher TF is 12x primary (5m -> 1h, 15m -> 3h)
        
        # Order Configuration
        self.order_type = 'limit'  # 'market' or 'limit'
        self.limit_offset_pct = 0.05  # % offset for limit orders (0% maker fee!)
        self.limit_timeout = 30  # Seconds to wait for limit order fill
        
        # Limit order retry configuration (smart retry system)
        self.limit_retry_attempts = 5        # Number of retry attempts with price updates
        self.limit_retry_interval = 45       # Seconds between retries
        self.limit_skip_on_failure = True    # Skip trade if not filled (avoid taker fee)
        
        # Risk Management
        self.max_position_pct = 10.0  # % of balance
        self.stop_loss_pct = 1.2  # Tightened from 2.0 based on performance analysis
        self.take_profit_pct = 1.5  # Reduced from 3.0 - tighter target with 20x leverage
        self.max_daily_loss_pct = 5.0
        self.trailing_stop_pct = 0.8  # More aggressive trailing (from 1.0)
        self.trailing_activation_pct = 0.5  # Activate after 0.5% profit
        
        # Direction-based confidence thresholds (from performance analysis)
        # LONG positions were losing (-0.17 USDT), SHORT winning (+1.15 USDT)
        self.min_confidence_long = 65.0   # Higher bar for LONG (worse performance)
        self.min_confidence_short = 55.0  # Lower bar for SHORT (better performance)
        self.min_confidence_default = 60.0  # Fallback for other signals
        
        # Indicator Settings
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.ema_fast = 9
        self.ema_slow = 21
        self.volume_threshold = 1.5
        
        # ML Model Configuration
        self.ml_model_type = 'ensemble'  # 'ensemble', 'pytorch_lstm', 'pytorch_transformer'
        self.use_pytorch = False  # Enable PyTorch models (requires GPU for best performance)
    
    def calculate_higher_timeframe(self) -> str:
        """
        Calculate higher timeframe based on primary timeframe
        
        Returns:
            Higher timeframe string (e.g., '1h' for '5m')
        """
        # Timeframe to minutes mapping
        tf_to_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080
        }
        
        minutes_to_tf = {v: k for k, v in tf_to_minutes.items()}
        
        # Get primary timeframe in minutes
        primary_minutes = tf_to_minutes.get(self.timeframe, 5)
        
        # Calculate higher timeframe (12x multiplier by default)
        higher_minutes = primary_minutes * self.htf_multiplier
        
        # Find closest available timeframe
        available_minutes = sorted(tf_to_minutes.values())
        
        # Find the closest higher or equal timeframe
        for mins in available_minutes:
            if mins >= higher_minutes:
                return minutes_to_tf[mins]
        
        # If no higher timeframe, use weekly
        return '1w'
    
    def save_config_cache(self):
        """Save current configuration to cache file for quick restart"""
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'trading_pair': self.trading_pair,
            'timeframe': self.timeframe,
            'leverage': self.leverage,
            'margin_type': self.margin_type,
            'max_position_pct': self.max_position_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'order_type': self.order_type,
            'testnet': self.testnet
        }
        
        try:
            with open(self.CONFIG_CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            # Silently fail - cache is optional
            pass
    
    def load_config_cache(self) -> Optional[Dict]:
        """Load cached configuration if exists"""
        if not os.path.exists(self.CONFIG_CACHE_FILE):
            return None
        
        try:
            with open(self.CONFIG_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            return None
    
    def apply_cached_config(self, cache_data: Dict):
        """Apply cached configuration values"""
        self.trading_pair = cache_data.get('trading_pair', 'BTCUSDT')
        self.timeframe = cache_data.get('timeframe', '5m')
        self.leverage = cache_data.get('leverage', 10)
        self.margin_type = cache_data.get('margin_type', 'ISOLATED')
        self.max_position_pct = cache_data.get('max_position_pct', 10.0)
        self.stop_loss_pct = cache_data.get('stop_loss_pct', 2.0)
        self.take_profit_pct = cache_data.get('take_profit_pct', 3.0)
        self.order_type = cache_data.get('order_type', 'limit')
        # Note: testnet is always from .env, not from cache (security)
    
    def get_from_user(self):
        """Get configuration from user via CLI"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}═══ FUTURES BOT CONFIGURATION ═══{Colors.END}\n")
        
        # Check for cached configuration
        cached_config = self.load_config_cache()
        if cached_config:
            print(f"{Colors.GREEN}✓ Found saved configuration from previous session{Colors.END}")
            print(f"{Colors.CYAN}  Last used: {cached_config.get('timestamp', 'Unknown')}{Colors.END}\n")
            
            # Display cached settings
            print(f"{Colors.BOLD}Previous Settings:{Colors.END}")
            print(f"  • Trading Pair: {cached_config.get('trading_pair')}")
            print(f"  • Timeframe: {cached_config.get('timeframe')}")
            print(f"  • Leverage: {cached_config.get('leverage')}x")
            print(f"  • Position Size: {cached_config.get('max_position_pct')}%")
            print(f"  • Stop Loss: {cached_config.get('stop_loss_pct')}%")
            print(f"  • Take Profit: {cached_config.get('take_profit_pct')}%")
            print(f"  • Order Type: {cached_config.get('order_type')}")
            
            # Ask if user wants to use cached config
            use_cached = input(f"\n{Colors.CYAN}Use these settings? [Y/n]: {Colors.END}").strip().lower()
            
            if use_cached != 'n':
                # Apply cached config
                self.apply_cached_config(cached_config)
                
                # Display summary
                self._display_summary()
                
                # Final confirmation
                confirm = input(f"\n{Colors.CYAN}Start bot? [Y/n]: {Colors.END}").strip().lower()
                if confirm == 'n':
                    print(f"{Colors.YELLOW}Cancelled.{Colors.END}")
                    return False
                
                # Save config again (updates timestamp)
                self.save_config_cache()
                return True
            else:
                print(f"{Colors.CYAN}OK, let's configure from scratch...{Colors.END}\n")
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}═══ FUTURES BOT CONFIGURATION ═══{Colors.END}\n")
        
        # Check API keys
        if self.api_key and self.api_secret:
            print(f"{Colors.GREEN}✓ API credentials loaded from .env{Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠ No API credentials in .env{Colors.END}")
            self.api_key = input(f"{Colors.CYAN}API Key: {Colors.END}").strip()
            self.api_secret = input(f"{Colors.CYAN}API Secret: {Colors.END}").strip()
        
        # Trading pair
        self.trading_pair = input(f"\n{Colors.CYAN}Trading Pair [BTCUSDT]: {Colors.END}").strip().upper() or "BTCUSDT"
        
        # Strategy info (no choice needed - single expert system)
        print(f"\n{Colors.GREEN}✓ Strategy: Expert System (Experta-powered) 🧠{Colors.END}")
        print(f"  • 20+ trading rules with explainable decisions")
        print(f"  • Auto-adapts to TRENDING, RANGING, VOLATILE markets")
        print(f"  • Multi-indicator confluence with priority system")
        
        # Timeframe
        self.timeframe = input(f"{Colors.CYAN}Timeframe [5m]: {Colors.END}").strip().lower() or "5m"
        
        # Leverage
        leverage_input = input(f"{Colors.CYAN}Leverage [10]: {Colors.END}").strip()
        self.leverage = int(leverage_input) if leverage_input else 10
        
        # Risk settings
        print(f"\n{Colors.BOLD}Risk Management:{Colors.END}")
        pos_input = input(f"{Colors.CYAN}Max position % of balance [10]: {Colors.END}").strip()
        self.max_position_pct = float(pos_input) if pos_input else 10.0
        
        sl_input = input(f"{Colors.CYAN}Stop Loss % [2.0]: {Colors.END}").strip()
        self.stop_loss_pct = float(sl_input) if sl_input else 2.0
        
        tp_input = input(f"{Colors.CYAN}Take Profit % [3.0]: {Colors.END}").strip()
        self.take_profit_pct = float(tp_input) if tp_input else 3.0
        
        # Order type (for 0% maker fee on USDC pairs!)
        print(f"\n{Colors.BOLD}Order Type:{Colors.END}")
        print(f"  {Colors.GREEN}1. Limit Orders (0% maker fee on USDC!) ⭐ RECOMMENDED{Colors.END}")
        print("  2. Market Orders (fast execution, 0.04% taker fee)")
        order_choice = input(f"{Colors.CYAN}Select [1]: {Colors.END}").strip() or "1"
        self.order_type = 'limit' if order_choice == '1' else 'market'
        
        if self.order_type == 'limit':
            print(f"{Colors.GREEN}✓ Using limit orders - 0% fee on USDC pairs!{Colors.END}")
        
        # Testnet mode
        print(f"\n{Colors.BOLD}Trading Mode:{Colors.END}")
        testnet_input = input(f"{Colors.CYAN}Use Testnet? [Y/n]: {Colors.END}").strip().lower()
        self.testnet = testnet_input != 'n'
        
        # Confirm LIVE trading
        if not self.testnet:
            print(f"\n{Colors.RED}{Colors.BOLD}⚠ WARNING: LIVE TRADING WITH REAL MONEY!{Colors.END}")
            confirm = input(f"{Colors.RED}Type 'LIVE' to confirm: {Colors.END}").strip()
            if confirm != 'LIVE':
                print(f"{Colors.YELLOW}Cancelled. Using testnet instead.{Colors.END}")
                self.testnet = True
        
        # Display summary
        self._display_summary()
        
        # Final confirmation
        confirm = input(f"\n{Colors.CYAN}Start bot? [Y/n]: {Colors.END}").strip().lower()
        if confirm == 'n':
            print(f"{Colors.YELLOW}Cancelled.{Colors.END}")
            return False
        
        # Save configuration for quick restart next time
        self.save_config_cache()
        print(f"{Colors.GREEN}✓ Configuration saved for next restart{Colors.END}")
        
        return True
    
    def _display_summary(self):
        """Display configuration summary"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}═══ CONFIGURATION SUMMARY ═══{Colors.END}")
        print(f"{Colors.BOLD}Trading Pair:{Colors.END} {self.trading_pair}")
        print(f"{Colors.BOLD}Strategy:{Colors.END} Expert System (Experta)")
        print(f"{Colors.BOLD}Timeframe:{Colors.END} {self.timeframe}")
        print(f"{Colors.BOLD}Leverage:{Colors.END} {self.leverage}x {self.margin_type}")
        
        order_type_display = f"{Colors.GREEN}Limit (0% maker fee!){Colors.END}" if self.order_type == 'limit' else "Market (0.04% taker fee)"
        print(f"{Colors.BOLD}Order Type:{Colors.END} {order_type_display}")
        
        print(f"\n{Colors.BOLD}Risk:{Colors.END}")
        print(f"  • Position Size: {self.max_position_pct}%")
        print(f"  • Stop Loss: {self.stop_loss_pct}%")
        print(f"  • Take Profit: {self.take_profit_pct}%")
        print(f"  • Max Daily Loss: {self.max_daily_loss_pct}%")
        
        mode = f"{Colors.GREEN}🧪 TESTNET{Colors.END}" if self.testnet else f"{Colors.RED}🔴 LIVE{Colors.END}"
        print(f"\n{Colors.BOLD}Mode:{Colors.END} {mode}")
        print("═" * 40)
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.api_key or not self.api_secret:
            print(f"{Colors.RED}✗ Missing API credentials{Colors.END}")
            return False
        
        if self.leverage < 1 or self.leverage > 125:
            print(f"{Colors.RED}✗ Leverage must be between 1-125{Colors.END}")
            return False
        
        if self.max_position_pct <= 0 or self.max_position_pct > 100:
            print(f"{Colors.RED}✗ Position size must be 0-100%{Colors.END}")
            return False
        
        return True

