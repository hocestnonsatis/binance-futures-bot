#!/usr/bin/env python3
"""
Backtesting Tool for Hybrid Strategy
Tests strategy performance on historical data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from dotenv import load_dotenv

from binance_futures import BinanceFutures
from strategy import Strategy
from ml_model import HybridStrategy, MLSignalEnhancer
from indicators import TechnicalIndicators
from config import Config

load_dotenv()


class Backtester:
    """
    Backtest trading strategy on historical data
    
    Features:
    - Simulates realistic trading conditions
    - Calculates P&L, win rate, drawdown
    - Supports transaction fees
    - Detailed trade log
    """
    
    def __init__(self, config: Config, strategy, initial_capital: float = 10000):
        self.config = config
        self.strategy = strategy
        self.initial_capital = initial_capital
        
        # Backtest state
        self.capital = initial_capital
        self.position = None  # None, or {'side': 'LONG/SHORT', 'entry_price': float, 'quantity': float}
        self.trades = []
        self.equity_curve = []
        
    def run(self, df: pd.DataFrame, fee_rate: float = 0.0004):
        """
        Run backtest on historical data
        
        Args:
            df: Historical OHLCV data with indicators
            fee_rate: Trading fee rate (0.04% default for Binance Futures)
        
        Returns:
            Backtest results dictionary
        """
        print("\n" + "=" * 70)
        print("BACKTESTING STRATEGY")
        print("=" * 70)
        print(f"\n📊 Data: {len(df)} candles")
        print(f"   Period: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
        print(f"   Initial Capital: ${self.initial_capital:.2f}")
        print(f"   Fee Rate: {fee_rate * 100:.2f}%")
        print()
        
        # Need at least 200 candles for indicators
        if len(df) < 200:
            print("❌ Need at least 200 candles for backtesting")
            return None
        
        # Reset state
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        
        # Simulate trading
        print("🔄 Simulating trades...\n")
        
        for i in range(200, len(df)):
            # Get data up to current candle
            df_slice = df.iloc[:i+1]
            current_candle = df.iloc[i]
            
            current_price = current_candle['close']
            current_time = current_candle['timestamp']
            
            # Get strategy signal
            signal = self.strategy.analyze(df_slice)
            
            # Execute trades based on signal
            if self.position is None:
                # No position - check for entry
                if signal['signal'] in ['BUY', 'SELL'] and signal['confidence'] >= 60:
                    # Open position
                    position_size = self.capital * 0.95  # Use 95% of capital
                    quantity = position_size / current_price
                    
                    # Calculate fees
                    entry_fee = position_size * fee_rate
                    self.capital -= entry_fee
                    
                    self.position = {
                        'side': 'LONG' if signal['signal'] == 'BUY' else 'SHORT',
                        'entry_price': current_price,
                        'quantity': quantity,
                        'entry_time': current_time,
                        'entry_capital': self.capital,
                        'signal_confidence': signal['confidence'],
                        'reasons': signal['reasons'][:3],  # Top 3 reasons
                        'entry_fee': entry_fee
                    }
            
            else:
                # Have position - check for exit
                should_exit = False
                exit_reason = ""
                
                # Exit on opposite signal
                if self.position['side'] == 'LONG' and signal['signal'] == 'SELL' and signal['confidence'] >= 55:
                    should_exit = True
                    exit_reason = "Opposite signal (SELL)"
                elif self.position['side'] == 'SHORT' and signal['signal'] == 'BUY' and signal['confidence'] >= 55:
                    should_exit = True
                    exit_reason = "Opposite signal (BUY)"
                
                # Simple stop loss (10%) and take profit (20%)
                pnl_pct = 0
                if self.position['side'] == 'LONG':
                    pnl_pct = ((current_price - self.position['entry_price']) / self.position['entry_price']) * 100
                else:  # SHORT
                    pnl_pct = ((self.position['entry_price'] - current_price) / self.position['entry_price']) * 100
                
                if pnl_pct <= -10:
                    should_exit = True
                    exit_reason = f"Stop loss ({pnl_pct:.1f}%)"
                elif pnl_pct >= 20:
                    should_exit = True
                    exit_reason = f"Take profit ({pnl_pct:.1f}%)"
                
                if should_exit:
                    # Close position
                    position_value = self.position['quantity'] * current_price
                    
                    # Calculate P&L
                    if self.position['side'] == 'LONG':
                        pnl = position_value - (self.position['quantity'] * self.position['entry_price'])
                    else:  # SHORT
                        pnl = (self.position['quantity'] * self.position['entry_price']) - position_value
                    
                    # Calculate fees
                    exit_fee = position_value * fee_rate
                    pnl -= exit_fee
                    pnl -= self.position['entry_fee']  # Total fees
                    
                    # Update capital
                    self.capital += pnl
                    
                    # Record trade
                    trade = {
                        'entry_time': self.position['entry_time'],
                        'exit_time': current_time,
                        'side': self.position['side'],
                        'entry_price': self.position['entry_price'],
                        'exit_price': current_price,
                        'quantity': self.position['quantity'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'confidence': self.position['signal_confidence'],
                        'exit_reason': exit_reason,
                        'capital_after': self.capital
                    }
                    
                    self.trades.append(trade)
                    
                    # Clear position
                    self.position = None
            
            # Track equity
            current_equity = self.capital
            if self.position is not None:
                # Add unrealized P&L
                position_value = self.position['quantity'] * current_price
                if self.position['side'] == 'LONG':
                    unrealized_pnl = position_value - (self.position['quantity'] * self.position['entry_price'])
                else:
                    unrealized_pnl = (self.position['quantity'] * self.position['entry_price']) - position_value
                
                current_equity += unrealized_pnl
            
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': current_equity
            })
        
        # Close any remaining position at last price
        if self.position is not None:
            print("⚠ Closing remaining position at last price")
            current_price = df.iloc[-1]['close']
            position_value = self.position['quantity'] * current_price
            
            if self.position['side'] == 'LONG':
                pnl = position_value - (self.position['quantity'] * self.position['entry_price'])
            else:
                pnl = (self.position['quantity'] * self.position['entry_price']) - position_value
            
            exit_fee = position_value * fee_rate
            pnl -= exit_fee
            pnl -= self.position['entry_fee']
            
            self.capital += pnl
            
            pnl_pct = (pnl / (self.position['quantity'] * self.position['entry_price'])) * 100
            
            trade = {
                'entry_time': self.position['entry_time'],
                'exit_time': df.iloc[-1]['timestamp'],
                'side': self.position['side'],
                'entry_price': self.position['entry_price'],
                'exit_price': current_price,
                'quantity': self.position['quantity'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'confidence': self.position['signal_confidence'],
                'exit_reason': 'End of backtest',
                'capital_after': self.capital
            }
            
            self.trades.append(trade)
            self.position = None
        
        # Calculate results
        return self._calculate_results()
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest performance metrics"""
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'message': 'No trades executed'
            }
        
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = self.capital - self.initial_capital
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = trades_df['pnl_pct'].values
        sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0
        
        # Maximum drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_capital': self.capital,
            'trades': trades_df,
            'equity_curve': equity_df
        }
    
    def print_results(self, results: Dict):
        """Print backtest results"""
        if results.get('total_trades', 0) == 0:
            print("❌ No trades executed during backtest")
            return
        
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print()
        
        # Performance metrics
        print("📊 Performance:")
        print(f"   Initial Capital:  ${self.initial_capital:.2f}")
        print(f"   Final Capital:    ${results['final_capital']:.2f}")
        print(f"   Total P&L:        ${results['total_pnl']:+.2f} ({results['total_return_pct']:+.2f}%)")
        print()
        
        # Trade statistics
        print("📈 Trade Statistics:")
        print(f"   Total Trades:     {results['total_trades']}")
        print(f"   Winning Trades:   {results['winning_trades']} ({results['win_rate']:.1f}%)")
        print(f"   Losing Trades:    {results['losing_trades']}")
        print(f"   Average Win:      ${results['avg_win']:.2f}")
        print(f"   Average Loss:     ${results['avg_loss']:.2f}")
        print(f"   Profit Factor:    {results['profit_factor']:.2f}")
        print()
        
        # Risk metrics
        print("⚠️  Risk Metrics:")
        print(f"   Max Drawdown:     {results['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        print()
        
        # Recent trades (last 5)
        print("📝 Last 5 Trades:")
        trades_df = results['trades']
        for _, trade in trades_df.tail(5).iterrows():
            pnl_symbol = "+" if trade['pnl'] > 0 else ""
            print(f"   {trade['side']:5} | Entry: ${trade['entry_price']:8.2f} | Exit: ${trade['exit_price']:8.2f} | "
                  f"P&L: {pnl_symbol}${trade['pnl']:7.2f} ({pnl_symbol}{trade['pnl_pct']:+.1f}%) | {trade['exit_reason']}")
        print()
        
        print("=" * 70)
        print()


def main():
    """Main backtesting function"""
    print("=" * 70)
    print("STRATEGY BACKTESTING TOOL")
    print("=" * 70)
    print()
    
    # Get parameters
    symbol = input("Trading symbol [BTCUSDT]: ").strip().upper() or 'BTCUSDT'
    timeframe = input("Timeframe [5m]: ").strip().lower() or '5m'
    days = int(input("Days of history [7]: ").strip() or '7')
    
    print()
    
    # Create config
    config = Config()
    config.trading_pair = symbol
    config.timeframe = timeframe
    
    # Fetch historical data
    print(f"📥 Fetching {days} days of {symbol} {timeframe} data...")
    
    try:
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        testnet = os.getenv('TESTNET', 'true').lower() == 'true'
        
        binance = BinanceFutures(api_key, api_secret, testnet)
        
        # Calculate how many candles we need
        minutes_per_candle = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440
        }
        
        candles_needed = min((days * 1440) // minutes_per_candle.get(timeframe, 5), 1500)
        
        df = binance.get_klines(symbol, timeframe, limit=candles_needed)
        print(f"✓ Fetched {len(df)} candles")
        
        # Calculate indicators
        print("🔄 Calculating indicators...")
        df = TechnicalIndicators.add_all_indicators(df)
        print("✓ Indicators ready")
        
        # Create strategy
        print(f"\n🎯 Creating strategy...")
        
        # Ask which strategy to test
        print("Strategy options:")
        print("  1. Rules Only (Experta)")
        print("  2. Hybrid (Experta + ML)")
        
        strategy_choice = input("Select strategy [2]: ").strip() or '2'
        
        experta_strategy = Strategy(config)
        
        if strategy_choice == '1':
            # Rules only
            strategy = experta_strategy
            strategy_name = "Experta Rules Only"
        else:
            # Hybrid
            strategy = HybridStrategy(config, experta_strategy)
            strategy_name = strategy.get_name()
        
        print(f"✓ Strategy: {strategy_name}")
        
        # Run backtest
        backtester = Backtester(config, strategy, initial_capital=10000)
        results = backtester.run(df)
        
        if results and results['total_trades'] > 0:
            backtester.print_results(results)
            
            # Save results
            save = input("Save results to CSV? [y/N]: ").strip().lower()
            if save == 'y':
                filename = f"backtest_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                results['trades'].to_csv(filename, index=False)
                print(f"✓ Results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

