"""
Configuration Management for Futures Trading Bot
Simple, clean configuration with CLI input and .env support
"""

import os
from dotenv import load_dotenv
from typing import Dict

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
    
    def __init__(self):
        load_dotenv()
        
        # API Configuration (from .env)
        self.api_key = os.getenv('BINANCE_API_KEY', '')
        self.api_secret = os.getenv('BINANCE_API_SECRET', '')
        self.testnet = os.getenv('TESTNET', 'true').lower() == 'true'
        
        # Trading Configuration (from CLI)
        self.trading_pair = ''
        self.timeframe = '5m'
        self.leverage = 10
        self.margin_type = 'ISOLATED'
        
        # Order Configuration
        self.order_type = 'limit'  # 'market' or 'limit'
        self.limit_offset_pct = 0.05  # % offset for limit orders (0% maker fee!)
        self.limit_timeout = 30  # Seconds to wait for limit order fill
        
        # Risk Management
        self.max_position_pct = 10.0  # % of balance
        self.stop_loss_pct = 2.0
        self.take_profit_pct = 3.0
        self.max_daily_loss_pct = 5.0
        
        # Indicator Settings
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.ema_fast = 9
        self.ema_slow = 21
        self.volume_threshold = 1.5
    
    def get_from_user(self):
        """Get configuration from user via CLI"""
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

