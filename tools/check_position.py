#!/usr/bin/env python3
"""
Check Current Position Status
Quick script to see position details and why it's not closing
"""

import os
import sys
from datetime import datetime
from binance_futures import BinanceFutures
from database import Database
from config import Colors
from dotenv import load_dotenv

load_dotenv()


def main():
    print(f"\n{Colors.CYAN}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}POSITION STATUS CHECKER{Colors.END}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.END}\n")
    
    # Get credentials
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    testnet = os.getenv('TESTNET', 'true').lower() == 'true'
    
    if not api_key or not api_secret:
        print(f"{Colors.RED}❌ API credentials not found in .env{Colors.END}")
        return
    
    # Connect
    mode = "TESTNET" if testnet else "LIVE"
    print(f"{Colors.CYAN}Connecting to Binance {mode}...{Colors.END}")
    
    try:
        binance = BinanceFutures(api_key, api_secret, testnet)
        db = Database('bot.db')
        print(f"{Colors.GREEN}✓ Connected{Colors.END}\n")
    except Exception as e:
        print(f"{Colors.RED}❌ Connection failed: {e}{Colors.END}")
        return
    
    # Get trading pair from config or default
    symbol = os.getenv('TRADING_PAIR', 'BTCUSDT')
    
    # Check exchange position
    print(f"{Colors.BOLD}BINANCE POSITION:{Colors.END}")
    print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
    
    try:
        position = binance.get_position(symbol)
        
        if position:
            entry_price = position['entry_price']
            quantity = position['quantity']
            side = position['side']
            
            # Get current price
            ticker = binance.client.futures_mark_price(symbol=symbol)
            current_price = float(ticker['markPrice'])
            
            # Calculate P&L
            if side in ['LONG', 'BUY']:
                pnl_per_unit = current_price - entry_price
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl_per_unit = entry_price - current_price
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            pnl = pnl_per_unit * quantity
            
            color = Colors.GREEN if pnl > 0 else Colors.RED
            
            print(f"Symbol:       {symbol}")
            print(f"Side:         {Colors.BOLD}{side}{Colors.END}")
            print(f"Entry Price:  ${entry_price:,.2f}")
            print(f"Current Price: ${current_price:,.2f}")
            print(f"Quantity:     {quantity:.6f}")
            print(f"P&L:          {color}{pnl:+.2f} USDT ({pnl_pct:+.2f}%){Colors.END}")
            print()
            
            # Calculate exit levels (assuming default config)
            stop_loss_pct = 2.0
            take_profit_pct = 3.0
            
            if side in ['LONG', 'BUY']:
                stop_loss = entry_price * (1 - stop_loss_pct / 100)
                take_profit = entry_price * (1 + take_profit_pct / 100)
            else:  # SHORT
                stop_loss = entry_price * (1 + stop_loss_pct / 100)
                take_profit = entry_price * (1 - take_profit_pct / 100)
            
            print(f"{Colors.BOLD}EXIT LEVELS:{Colors.END}")
            print(f"Stop Loss:    ${stop_loss:,.2f} ({'-' if side in ['LONG', 'BUY'] else '+'}2%)")
            print(f"Take Profit:  ${take_profit:,.2f} ({'+' if side in ['LONG', 'BUY'] else '-'}3%)")
            print()
            
            # Check trailing stop
            if side in ['LONG', 'BUY']:
                # For LONG, we'd track highest price
                trailing_stop = current_price * (1 - stop_loss_pct / 100)
                print(f"{Colors.CYAN}Trailing Stop: ${trailing_stop:,.2f} (follows price up){Colors.END}")
            else:  # SHORT
                # For SHORT, we track lowest price
                trailing_stop = current_price * (1 + stop_loss_pct / 100)
                print(f"{Colors.CYAN}Trailing Stop: ${trailing_stop:,.2f} (follows price down){Colors.END}")
            
            print()
            
            # Analysis
            print(f"{Colors.BOLD}ANALYSIS:{Colors.END}")
            print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
            
            if side == 'SHORT' or side == 'SELL':
                print(f"📉 SHORT Position Analysis:")
                print()
                
                if pnl > 0:
                    print(f"  ✅ Position is PROFITABLE (+{pnl:.2f} USDT)")
                    print()
                    print(f"  Position will close when:")
                    print(f"    1. Price rises to ${trailing_stop:.2f} (trailing stop)")
                    print(f"    2. Profit reaches {take_profit_pct}% (${take_profit:,.2f})")
                    print(f"    3. BUY signal with 50%+ confidence")
                    print()
                    
                    if pnl_pct > 2:
                        print(f"  {Colors.GREEN}💡 Good profit! Consider taking profit manually{Colors.END}")
                    else:
                        print(f"  {Colors.YELLOW}💡 Let it run, trailing stop will protect profit{Colors.END}")
                else:
                    print(f"  ⚠️  Position is LOSING ({pnl:.2f} USDT)")
                    print()
                    
                    distance_to_sl = ((stop_loss - current_price) / current_price) * 100
                    print(f"  Distance to stop loss: {distance_to_sl:.2f}%")
                    
                    if abs(pnl_pct) > 1.5:
                        print(f"  {Colors.RED}⚠️  Consider closing manually to cut losses{Colors.END}")
            
            else:  # LONG
                print(f"📈 LONG Position Analysis:")
                print()
                
                if pnl > 0:
                    print(f"  ✅ Position is PROFITABLE (+{pnl:.2f} USDT)")
                    print()
                    print(f"  Position will close when:")
                    print(f"    1. Price drops to ${trailing_stop:.2f} (trailing stop)")
                    print(f"    2. Profit reaches {take_profit_pct}% (${take_profit:,.2f})")
                    print(f"    3. SELL signal with 50%+ confidence")
                else:
                    print(f"  ⚠️  Position is LOSING ({pnl:.2f} USDT)")
            
            print()
            
        else:
            print(f"{Colors.GREEN}No open position on Binance{Colors.END}\n")
            
    except Exception as e:
        print(f"{Colors.RED}❌ Error checking Binance position: {e}{Colors.END}\n")
    
    # Check database
    print(f"{Colors.BOLD}DATABASE POSITION:{Colors.END}")
    print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
    
    try:
        db_position = db.get_open_position(symbol)
        
        if db_position:
            print(f"Symbol:       {db_position['symbol']}")
            print(f"Side:         {db_position['side']}")
            print(f"Entry Price:  ${db_position['entry_price']:,.2f}")
            print(f"Quantity:     {db_position['quantity']:.6f}")
            print(f"Stop Loss:    ${db_position.get('stop_loss', 0):,.2f}")
            print(f"Take Profit:  ${db_position.get('take_profit', 0):,.2f}")
            print()
        else:
            print(f"{Colors.GREEN}No open position in database{Colors.END}\n")
            
    except Exception as e:
        print(f"{Colors.RED}❌ Error checking database: {e}{Colors.END}\n")
    
    # Get recent logs
    print(f"{Colors.BOLD}RECENT LOGS (Last 10):{Colors.END}")
    print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
    
    try:
        import sqlite3
        conn = sqlite3.connect('bot.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, level, message 
            FROM logs 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        
        logs = cursor.fetchall()
        
        for log in reversed(logs):
            timestamp, level, message = log
            
            if level == 'ERROR':
                color = Colors.RED
            elif level == 'WARNING':
                color = Colors.YELLOW
            elif level == 'INFO':
                color = Colors.GREEN
            else:
                color = Colors.CYAN
            
            # Shorten timestamp
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime('%H:%M:%S')
            
            print(f"{color}[{time_str}] {level:7s}: {message}{Colors.END}")
        
        conn.close()
        print()
        
    except Exception as e:
        print(f"{Colors.YELLOW}⚠ Could not read logs: {e}{Colors.END}\n")
    
    # Options
    print(f"{Colors.BOLD}OPTIONS:{Colors.END}")
    print(f"{Colors.CYAN}{'-' * 70}{Colors.END}")
    print()
    print(f"1. {Colors.GREEN}Let bot manage it{Colors.END} - Trailing stop will close when profit secured")
    print(f"2. {Colors.YELLOW}Close manually{Colors.END} - Use Binance app/website")
    print(f"3. {Colors.CYAN}Adjust config{Colors.END} - Change take_profit_pct or stop_loss_pct")
    print()
    
    print(f"{Colors.CYAN}{'=' * 70}{Colors.END}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()

