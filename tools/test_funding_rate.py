#!/usr/bin/env python3
"""
Test Funding Rate Information
Shows current funding rate and calculates holding costs
"""

import os
from datetime import datetime
from binance_futures import BinanceFutures
from dotenv import load_dotenv

# Load environment
load_dotenv()


def format_timestamp(ts_ms):
    """Format timestamp to readable date"""
    return datetime.fromtimestamp(ts_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')


def calculate_holding_costs(position_size, funding_rate_pct, days):
    """Calculate cost of holding position"""
    # 3 funding payments per day (every 8 hours)
    payments_per_day = 3
    total_payments = days * payments_per_day
    
    # Cost per payment
    cost_per_payment = position_size * (funding_rate_pct / 100)
    
    # Total cost
    total_cost = cost_per_payment * total_payments
    
    return {
        'daily_cost': cost_per_payment * payments_per_day,
        'weekly_cost': cost_per_payment * payments_per_day * 7,
        'monthly_cost': cost_per_payment * payments_per_day * 30,
        'cost_per_payment': cost_per_payment
    }


def main():
    print("=" * 70)
    print("BINANCE FUTURES FUNDING RATE CHECKER")
    print("=" * 70)
    print()
    
    # Get API credentials
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    testnet = os.getenv('TESTNET', 'true').lower() == 'true'
    
    if not api_key or not api_secret:
        print("❌ Error: BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env")
        return
    
    # Connect to Binance
    mode = "TESTNET" if testnet else "LIVE"
    print(f"Connecting to Binance {mode}...")
    
    try:
        binance = BinanceFutures(api_key, api_secret, testnet)
        print(f"✓ Connected to Binance {mode}")
        print()
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Symbol to check
    symbol = 'BTCUSDT'
    
    # Get current funding rate
    print(f"📊 FUNDING RATE: {symbol}")
    print("-" * 70)
    
    try:
        funding_info = binance.get_funding_rate(symbol)
        
        funding_rate = funding_info['funding_rate']
        funding_rate_pct = funding_info['funding_rate_pct']
        next_funding = funding_info['next_funding_time']
        mark_price = funding_info['mark_price']
        
        print(f"Current Funding Rate: {funding_rate:.6f} ({funding_rate_pct:.4f}%)")
        print(f"Mark Price: ${mark_price:,.2f}")
        print(f"Next Funding Time: {format_timestamp(next_funding)}")
        print()
        
        # Determine who pays
        if funding_rate > 0:
            print(f"💰 Funding Direction: LONG positions pay SHORT positions")
            print(f"   (Market is bullish - more longs than shorts)")
        elif funding_rate < 0:
            print(f"💰 Funding Direction: SHORT positions pay LONG positions")
            print(f"   (Market is bearish - more shorts than longs)")
        else:
            print(f"💰 Funding Direction: Neutral (no payment)")
        
        print()
        
        # Calculate holding costs for example position
        print("=" * 70)
        print("HOLDING COST EXAMPLES")
        print("=" * 70)
        print()
        
        # Example 1: $1,000 position (10x leverage = $10,000 position size)
        position_value = 1000
        leverage = 10
        position_size = position_value * leverage
        
        print(f"Example 1: ${position_value:,} margin @ {leverage}x leverage")
        print(f"Position Size: ${position_size:,}")
        print()
        
        costs = calculate_holding_costs(position_size, abs(funding_rate_pct), 1)
        
        print(f"  Per Payment (every 8h): ${costs['cost_per_payment']:.2f}")
        print(f"  Daily Cost (3x):        ${costs['daily_cost']:.2f}")
        print(f"  Weekly Cost:            ${costs['weekly_cost']:.2f}")
        print(f"  Monthly Cost:           ${costs['monthly_cost']:.2f}")
        print()
        
        # Example 2: $5,000 position (5x leverage)
        position_value2 = 5000
        leverage2 = 5
        position_size2 = position_value2 * leverage2
        
        print(f"Example 2: ${position_value2:,} margin @ {leverage2}x leverage")
        print(f"Position Size: ${position_size2:,}")
        print()
        
        costs2 = calculate_holding_costs(position_size2, abs(funding_rate_pct), 1)
        
        print(f"  Per Payment (every 8h): ${costs2['cost_per_payment']:.2f}")
        print(f"  Daily Cost (3x):        ${costs2['daily_cost']:.2f}")
        print(f"  Weekly Cost:            ${costs2['weekly_cost']:.2f}")
        print(f"  Monthly Cost:           ${costs2['monthly_cost']:.2f}")
        print()
        
        # Get historical funding rates
        print("=" * 70)
        print("HISTORICAL FUNDING RATES (Last 10)")
        print("=" * 70)
        print()
        
        history = binance.get_funding_rate_history(symbol, limit=10)
        
        print(f"{'Time':<20} {'Funding Rate':<15} {'%':<10}")
        print("-" * 70)
        
        for record in reversed(history):  # Show newest first
            timestamp = format_timestamp(record['funding_time'])
            rate = record['funding_rate']
            pct = record['funding_rate_pct']
            direction = "LONG→SHORT" if rate > 0 else "SHORT→LONG" if rate < 0 else "NEUTRAL"
            print(f"{timestamp:<20} {rate:>14.6f} {pct:>9.4f}%  {direction}")
        
        print()
        
        # Calculate average
        avg_rate = sum(r['funding_rate_pct'] for r in history) / len(history)
        print(f"Average Funding Rate: {avg_rate:.4f}%")
        print()
        
        # Recommendations
        print("=" * 70)
        print("💡 RECOMMENDATIONS")
        print("=" * 70)
        print()
        
        if abs(funding_rate_pct) > 0.05:
            print("⚠️  HIGH FUNDING RATE!")
            print(f"   Consider closing positions before next funding time")
            print(f"   Or switch to opposite side to earn funding")
        elif abs(funding_rate_pct) > 0.03:
            print("⚠️  Moderate funding rate")
            print(f"   Watch for opportunities to switch sides")
        else:
            print("✅ Normal funding rate")
            print(f"   Holding costs are reasonable")
        
        print()
        
        if funding_rate > 0:
            print("💡 Strategy Tips:")
            print("   • LONG positions: Consider taking profits before funding time")
            print("   • SHORT positions: Earn funding by holding through funding time")
        elif funding_rate < 0:
            print("💡 Strategy Tips:")
            print("   • SHORT positions: Consider closing before funding time")
            print("   • LONG positions: Earn funding by holding through funding time")
        
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

