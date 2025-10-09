#!/usr/bin/env python3
"""
Binance Futures Trading Bot
Clean, minimal implementation with proper error handling
"""

import sys
import signal
import time
from datetime import datetime

# Import our modules
from config import Config, Colors
from database import Database
from binance_futures import BinanceFutures
from risk_manager import RiskManager
from indicators import TechnicalIndicators

# Import strategy (hybrid: expert system + ML)
from strategy import Strategy
from ml_model import HybridStrategy, MLSignalEnhancer


class FuturesBot:
    """Main trading bot class"""
    
    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.running = False
        self.binance = None
        self.risk_manager = None
        self.strategy = None
        self.quote_asset = None  # Will be set during initialization
        
        # Stats
        self.last_signal_time = None
        self.signals_processed = 0
        
        # Trailing stop state
        self.position_highest_price = None  # Track highest price for trailing stop
    
    def initialize(self):
        """Initialize bot components"""
        print(f"\n{Colors.CYAN}{'═' * 60}{Colors.END}")
        print(f"{Colors.BOLD}Initializing Futures Trading Bot{Colors.END}")
        print(f"{Colors.CYAN}{'═' * 60}{Colors.END}\n")
        
        # Initialize Binance client
        print(f"{Colors.CYAN}Connecting to Binance...{Colors.END}")
        try:
            self.binance = BinanceFutures(
                self.config.api_key,
                self.config.api_secret,
                self.config.testnet
            )
            
            # Test connection and get balance
            self.quote_asset = self.binance.get_quote_asset(self.config.trading_pair)
            balance = self.binance.get_balance(self.quote_asset)
            mode = "TESTNET" if self.config.testnet else "LIVE"
            print(f"{Colors.GREEN}✓ Connected to Binance {mode}{Colors.END}")
            print(f"{Colors.GREEN}✓ Trading Pair: {self.config.trading_pair} (Quote: {self.quote_asset}){Colors.END}")
            print(f"{Colors.GREEN}✓ Balance: {balance:.2f} {self.quote_asset}{Colors.END}")
            
            self.db.info(f"Connected to Binance {mode} - Balance: {balance:.2f} {self.quote_asset}")
        except Exception as e:
            print(f"{Colors.RED}✗ Failed to connect: {e}{Colors.END}")
            self.db.error(f"Connection failed: {e}")
            return False
        
        # Set position mode (One-Way Mode)
        print(f"{Colors.CYAN}Setting position mode...{Colors.END}")
        try:
            self.binance.set_position_mode(dual_side=False)  # One-Way Mode
            print(f"{Colors.GREEN}✓ Position mode: One-Way{Colors.END}")
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Position mode: {e}{Colors.END}")
        
        # Set leverage and margin type
        print(f"{Colors.CYAN}Setting up leverage...{Colors.END}")
        try:
            self.binance.set_leverage(self.config.trading_pair, self.config.leverage)
            self.binance.set_margin_type(self.config.trading_pair, self.config.margin_type)
            print(f"{Colors.GREEN}✓ Leverage: {self.config.leverage}x {self.config.margin_type}{Colors.END}")
            
            self.db.info(f"Set leverage {self.config.leverage}x {self.config.margin_type}")
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Leverage setup: {e}{Colors.END}")
        
        # Initialize risk manager
        self.risk_manager = RiskManager(self.config)
        print(f"{Colors.GREEN}✓ Risk manager initialized{Colors.END}")
        
        # Calculate higher timeframe for multi-timeframe analysis
        if self.config.use_multi_timeframe:
            self.config.higher_timeframe = self.config.calculate_higher_timeframe()
            print(f"{Colors.CYAN}✓ Multi-Timeframe Analysis: {self.config.timeframe} + {self.config.higher_timeframe}{Colors.END}")
        
        # Initialize strategy (hybrid: expert system + ML enhancement)
        experta_strategy = Strategy(self.config)
        
        # Create hybrid strategy (will auto-load symbol+timeframe specific model)
        self.strategy = HybridStrategy(self.config, experta_strategy)
        
        strategy_name = self.strategy.get_name()
        print(f"{Colors.GREEN}✓ Strategy: {strategy_name}{Colors.END}")
        
        # Check ML model status
        ml_info = self.strategy.ml_enhancer.get_model_info()
        
        if ml_info['is_trained']:
            # Model exists and loaded
            print(f"{Colors.CYAN}  🤖 ML Enhancement: Active{Colors.END}")
            if ml_info['symbol'] and ml_info['timeframe']:
                if ml_info.get('is_alternative'):
                    print(f"{Colors.CYAN}     Model: {ml_info['actual_model_symbol']} {ml_info['timeframe']} (similar to {ml_info['symbol']}){Colors.END}")
                else:
                    print(f"{Colors.CYAN}     Model: {ml_info['symbol']} {ml_info['timeframe']}{Colors.END}")
        else:
            # No model - automatically download data and train
            print(f"{Colors.YELLOW}  🤖 ML Enhancement: Not trained{Colors.END}")
            if ml_info['symbol'] and ml_info['timeframe']:
                print(f"{Colors.CYAN}     No model for: {ml_info['symbol']} {ml_info['timeframe']}{Colors.END}")
            
            # Automatically train model (no user confirmation needed)
            print(f"\n{Colors.CYAN}📥 Auto-training ML model...{Colors.END}")
            print(f"{Colors.CYAN}   Will check cached data or download from Binance (~30-120 seconds){Colors.END}")
            print()
            
            success = self.strategy.ml_enhancer.auto_train(self.binance, limit=2000)
            
            if success:
                print(f"{Colors.GREEN}✓ ML model trained successfully!{Colors.END}")
                print(f"{Colors.CYAN}  🤖 ML Enhancement: Now Active{Colors.END}")
                self.db.info(f"ML model auto-trained: {ml_info['symbol']} {ml_info['timeframe']}")
            else:
                print(f"{Colors.YELLOW}⚠ Auto-training failed - continuing with rules only{Colors.END}")
                print(f"{Colors.CYAN}  Bot will still work using Experta rules (70% of strategy){Colors.END}")
        
        self.db.info(f"Strategy initialized: {strategy_name}")
        
        print(f"\n{Colors.GREEN}{'═' * 60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}Bot Ready!{Colors.END}")
        print(f"{Colors.GREEN}{'═' * 60}{Colors.END}\n")
        
        return True
    
    def trading_loop(self):
        """Main trading loop"""
        print(f"{Colors.BOLD}Starting trading loop...{Colors.END}")
        print(f"{Colors.YELLOW}Press Ctrl+C to stop{Colors.END}\n")
        
        self.running = True
        loop_count = 0
        
        while self.running:
            try:
                loop_count += 1
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Get market data (primary timeframe)
                df = self.binance.get_klines(
                    self.config.trading_pair,
                    self.config.timeframe,
                    limit=200
                )
                
                # Calculate indicators for primary timeframe
                df = TechnicalIndicators.add_all_indicators(df)
                
                # Add higher timeframe indicators if enabled
                if self.config.use_multi_timeframe and self.config.higher_timeframe:
                    try:
                        # Fetch higher timeframe data
                        df_htf = self.binance.get_klines(
                            self.config.trading_pair,
                            self.config.higher_timeframe,
                            limit=200
                        )
                        
                        # Calculate indicators for higher timeframe
                        df_htf = TechnicalIndicators.add_all_indicators(df_htf)
                        
                        # Merge higher timeframe indicators into primary
                        htf_suffix = f"htf_{self.config.higher_timeframe}"
                        df = TechnicalIndicators.add_higher_timeframe_indicators(
                            df, df_htf, htf_suffix
                        )
                    except Exception as e:
                        # If HTF fetch fails, continue with primary TF only
                        if loop_count % 10 == 0:
                            print(f"{Colors.YELLOW}⚠ HTF data fetch failed: {e}{Colors.END}")
                
                # Get current price
                current_price = df.iloc[-1]['close']
                
                # Get account status
                balance = self.binance.get_balance(self.quote_asset)
                current_position = self.binance.get_position(self.config.trading_pair)
                daily_pnl = self.db.get_daily_pnl()
                
                # Check if we have an open position
                if current_position:
                    self._manage_position(current_position, current_price, current_time, df)
                else:
                    self._check_entry_signals(df, balance, daily_pnl, current_price, current_time)
                
                # Display status every 10 loops
                if loop_count % 10 == 0:
                    self._display_status(balance, current_price, current_position, daily_pnl)
                
                # Wait before next iteration
                time.sleep(5)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                error_msg = f"Error in trading loop: {e}"
                print(f"{Colors.RED}✗ {error_msg}{Colors.END}")
                self.db.error(error_msg)
                time.sleep(10)  # Wait longer on error
    
    def _check_entry_signals(self, df, balance, daily_pnl, current_price, current_time):
        """Check for entry signals"""
        # Validate if we can trade
        can_trade, reason = self.risk_manager.validate_trade(balance, None, daily_pnl)
        
        if not can_trade:
            if self.signals_processed % 20 == 0:  # Log occasionally
                print(f"{Colors.YELLOW}[{current_time}] {reason}{Colors.END}")
            self.signals_processed += 1
            return
        
        # Run strategy analysis
        signal = self.strategy.analyze(df)
        self.signals_processed += 1
        
        # Apply direction-specific confidence threshold
        if signal['signal'] == 'BUY':
            min_conf = self.config.min_confidence_long
        elif signal['signal'] == 'SELL':
            min_conf = self.config.min_confidence_short
        else:
            min_conf = self.config.min_confidence_default
        
        # Check confidence threshold
        if signal['signal'] in ['BUY', 'SELL'] and signal['confidence'] < min_conf:
            if self.signals_processed % 20 == 0:
                print(f"{Colors.YELLOW}[{current_time}] {signal['signal']} signal {signal['confidence']:.0f}% below threshold {min_conf:.0f}%{Colors.END}")
            return
        
        # Log signal
        signal_msg = f"Signal: {signal['signal']} ({signal['confidence']:.0f}%)"
        print(f"{Colors.CYAN}[{current_time}] {signal_msg}{Colors.END}")
        
        if signal['signal'] != 'HOLD':
            reasons_str = ", ".join(signal['reasons'][:3])  # Top 3 reasons
            print(f"  Reasons: {reasons_str}")
            self.db.debug(f"{signal_msg} - {reasons_str}")
        
        # Execute trade if signal is strong
        if signal['signal'] in ['BUY', 'SELL']:
            self._execute_entry(signal, current_price, balance, current_time)
    
    def _execute_entry_with_smart_limit(self, side, quantity):
        """
        Smart limit order with retries and price updates
        Returns order dict if successful, None if all attempts failed
        """
        max_attempts = self.config.limit_retry_attempts
        retry_interval = self.config.limit_retry_interval
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Get fresh current price for each attempt
                fresh_price = self.binance.get_current_price(self.config.trading_pair)
                
                # Calculate limit price with offset
                if side == 'BUY':
                    limit_price = fresh_price * (1 - self.config.limit_offset_pct / 100)
                else:  # SELL
                    limit_price = fresh_price * (1 + self.config.limit_offset_pct / 100)
                
                print(f"{Colors.CYAN}Attempt {attempt}/{max_attempts}: Limit @ {limit_price:.2f} (fresh price: {fresh_price:.2f}){Colors.END}")
                
                # Cancel previous order if exists
                if attempt > 1:
                    self.binance.cancel_all_orders(self.config.trading_pair)
                    time.sleep(0.5)  # Small delay after cancel
                
                # Place new limit order
                order = self.binance.place_limit_order(
                    self.config.trading_pair, side, quantity, limit_price
                )
                
                # Wait for fill
                filled = self._wait_for_order_fill(order['order_id'], retry_interval)
                
                if filled:
                    print(f"{Colors.GREEN}✓ Limit order filled at {limit_price:.2f}{Colors.END}")
                    return order  # Success!
                
                print(f"{Colors.YELLOW}Not filled, retrying with updated price...{Colors.END}")
                
            except Exception as e:
                print(f"{Colors.RED}Error on attempt {attempt}: {e}{Colors.END}")
                if attempt < max_attempts:
                    time.sleep(2)  # Brief pause before retry
        
        # All attempts failed
        if self.config.limit_skip_on_failure:
            print(f"{Colors.YELLOW}⚠ Failed after {max_attempts} attempts - skipping trade to avoid taker fee{Colors.END}")
            self.binance.cancel_all_orders(self.config.trading_pair)
            return None
        else:
            # Fallback to market order (not recommended - high fees)
            print(f"{Colors.YELLOW}⚠ Using market order as fallback{Colors.END}")
            return self.binance.place_market_order(self.config.trading_pair, side, quantity)
    
    def _execute_entry(self, signal, current_price, balance, current_time):
        """Execute entry trade with smart limit order retry"""
        try:
            # Calculate position size
            quantity = self.risk_manager.calculate_position_size(balance, current_price)
            
            # Determine side
            side = signal['signal']  # BUY or SELL
            
            print(f"\n{Colors.BOLD}{Colors.GREEN}{'═' * 60}{Colors.END}")
            print(f"{Colors.BOLD}OPENING {side} POSITION{Colors.END}")
            print(f"{Colors.GREEN}{'═' * 60}{Colors.END}")
            print(f"Price: {current_price:.2f} {self.quote_asset}")
            print(f"Quantity: {quantity:.6f}")
            print(f"Confidence: {signal['confidence']:.0f}%")
            
            # Place order based on config
            if self.config.order_type == 'limit':
                # Use smart limit order retry system
                order = self._execute_entry_with_smart_limit(side, quantity)
                
                if order is None:
                    # Trade skipped after all retries
                    print(f"{Colors.YELLOW}Trade skipped - limit order not filled{Colors.END}")
                    self.db.info(f"Skipped {side} trade - limit order timeout after {self.config.limit_retry_attempts} attempts")
                    return
            else:
                # Market order (fast but 0.04% taker fee)
                print(f"{Colors.CYAN}Placing market order (0.04% taker fee){Colors.END}")
                order = self.binance.place_market_order(
                    self.config.trading_pair,
                    side,
                    quantity
                )
            
            entry_price = order['price']
            actual_qty = order['quantity']
            
            # Calculate stop loss and take profit
            stop_loss = self.risk_manager.calculate_stop_loss_price(entry_price, side)
            take_profit = self.risk_manager.calculate_take_profit_price(entry_price, side)
            
            print(f"{Colors.GREEN}✓ Order filled at {entry_price:.2f}{Colors.END}")
            print(f"Initial Stop Loss: {stop_loss:.2f} ({self.config.stop_loss_pct}%)")
            print(f"Take Profit Target: {take_profit:.2f} (+{self.config.take_profit_pct}%)")
            print(f"{Colors.CYAN}🔄 Trailing Stop Active: Will track highest price{Colors.END}")
            print(f"{Colors.GREEN}{'═' * 60}{Colors.END}\n")
            
            # Initialize trailing stop at entry price
            self.position_highest_price = entry_price
            self.db.info(f"Trailing stop initialized at {entry_price:.2f}")
            
            # Record trade in database
            self.db.add_trade(
                symbol=self.config.trading_pair,
                side=side,
                price=entry_price,
                quantity=actual_qty,
                strategy=self.strategy.get_name(),
                order_id=str(order['order_id'])
            )
            
            # Record position
            self.db.add_position(
                symbol=self.config.trading_pair,
                side=side,
                entry_price=entry_price,
                quantity=actual_qty,
                leverage=self.config.leverage,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            log_msg = f"Opened {side} position: {actual_qty:.6f} @ {entry_price:.2f}"
            self.db.info(log_msg)
            
        except Exception as e:
            error_msg = f"Failed to execute entry: {e}"
            print(f"{Colors.RED}✗ {error_msg}{Colors.END}")
            self.db.error(error_msg)
    
    def _wait_for_order_fill(self, order_id: str, timeout: int) -> bool:
        """
        Wait for limit order to fill
        Returns True if filled, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check order status
                order_info = self.binance.client.futures_get_order(
                    symbol=self.config.trading_pair,
                    orderId=order_id
                )
                
                status = order_info['status']
                
                if status == 'FILLED':
                    return True
                elif status in ['CANCELED', 'REJECTED', 'EXPIRED']:
                    return False
                
                # Wait 1 second before checking again
                time.sleep(1)
                
            except Exception as e:
                self.db.error(f"Error checking order status: {e}")
                return False
        
        # Timeout
        return False
    
    def _manage_position(self, position, current_price, current_time, df):
        """Manage open position with trailing stop and strategy exit signals"""
        entry_price = position['entry_price']
        side = position['side']
        quantity = position['quantity']
        
        # Initialize highest price if not set
        if self.position_highest_price is None:
            self.position_highest_price = entry_price
        
        # Calculate current P&L
        pnl = self.risk_manager.calculate_pnl(
            entry_price,
            current_price,
            quantity,
            side,
            self.config.leverage
        )
        
        pnl_pct = (pnl / (entry_price * quantity)) * 100 * self.config.leverage
        
        # === CHECK 1: Strategy Exit Signal ===
        # Analyze current market conditions
        signal = self.strategy.analyze(df)
        
        # Check for reverse signal (exit signal)
        exit_confidence_threshold = 50  # Exit if reverse signal has 50%+ confidence
        
        if side in ['LONG', 'BUY']:
            # LONG position: exit on SELL signal
            if signal['signal'] == 'SELL' and signal['confidence'] >= exit_confidence_threshold:
                reason = f"Strategy exit: SELL signal ({signal['confidence']:.0f}%) - {', '.join(signal['reasons'][:2])}"
                print(f"{Colors.YELLOW}[{current_time}] 🔄 Trend reversal detected!{Colors.END}")
                self._close_position(position, current_price, pnl, reason, current_time)
                self.position_highest_price = None
                return
        else:
            # SHORT position: exit on BUY signal
            if signal['signal'] == 'BUY' and signal['confidence'] >= exit_confidence_threshold:
                reason = f"Strategy exit: BUY signal ({signal['confidence']:.0f}%) - {', '.join(signal['reasons'][:2])}"
                print(f"{Colors.YELLOW}[{current_time}] 🔄 Trend reversal detected!{Colors.END}")
                self._close_position(position, current_price, pnl, reason, current_time)
                self.position_highest_price = None
                return
        
        # === CHECK 2: Update Trailing Stop ===
        if side in ['LONG', 'BUY']:
            # For LONG: track highest price
            if current_price > self.position_highest_price:
                old_highest = self.position_highest_price
                self.position_highest_price = current_price
                trailing_stop = self.risk_manager.calculate_trailing_stop_price(
                    self.position_highest_price, side
                )
                print(f"{Colors.CYAN}[{current_time}] 📈 New high: {current_price:.2f} → Trailing stop: {trailing_stop:.2f}{Colors.END}")
        else:
            # For SHORT: track lowest price
            if current_price < self.position_highest_price or self.position_highest_price == entry_price:
                old_lowest = self.position_highest_price
                self.position_highest_price = current_price
                trailing_stop = self.risk_manager.calculate_trailing_stop_price(
                    self.position_highest_price, side
                )
                print(f"{Colors.CYAN}[{current_time}] 📉 New low: {current_price:.2f} → Trailing stop: {trailing_stop:.2f}{Colors.END}")
        
        # === CHECK 3: Trailing Stop Exit ===
        should_close, reason = self.risk_manager.should_close_position_trailing(
            position, 
            current_price, 
            self.position_highest_price
        )
        
        if should_close:
            self._close_position(position, current_price, pnl, reason, current_time)
            self.position_highest_price = None
        else:
            # Display status
            color = Colors.GREEN if pnl > 0 else Colors.RED
            trailing_stop = self.risk_manager.calculate_trailing_stop_price(
                self.position_highest_price, side
            )
            print(f"{color}[{current_time}] {side} Position: {pnl:+.2f} {self.quote_asset} ({pnl_pct:+.2f}%) | Trailing: {trailing_stop:.2f}{Colors.END}")
    
    def _close_position(self, position, current_price, pnl, reason, current_time):
        """Close position"""
        try:
            side = position['side']
            
            print(f"\n{Colors.BOLD}{Colors.YELLOW}{'═' * 60}{Colors.END}")
            print(f"{Colors.BOLD}CLOSING {side} POSITION{Colors.END}")
            print(f"{Colors.YELLOW}{'═' * 60}{Colors.END}")
            print(f"Reason: {reason}")
            print(f"P&L: {pnl:+.2f} {self.quote_asset}")
            
            # Close position
            order = self.binance.close_position(self.config.trading_pair)
            
            if order:
                exit_price = order['price']
                print(f"{Colors.GREEN}✓ Position closed at {exit_price:.2f}{Colors.END}")
                print(f"{Colors.YELLOW}{'═' * 60}{Colors.END}\n")
                
                # Record exit trade
                close_side = 'SELL' if side == 'LONG' else 'BUY'
                self.db.add_trade(
                    symbol=self.config.trading_pair,
                    side=close_side,
                    price=exit_price,
                    quantity=position['quantity'],
                    strategy=self.strategy.get_name(),
                    pnl=pnl,
                    order_id=str(order['order_id'])
                )
                
                # Update position in database
                db_position = self.db.get_open_position(self.config.trading_pair)
                if db_position:
                    self.db.close_position(db_position['id'], pnl)
                
                log_msg = f"Closed {side} position: {pnl:+.2f} {self.quote_asset} - {reason}"
                self.db.info(log_msg)
                
        except Exception as e:
            error_msg = f"Failed to close position: {e}"
            print(f"{Colors.RED}✗ {error_msg}{Colors.END}")
            self.db.error(error_msg)
    
    def _display_status(self, balance, price, position, daily_pnl):
        """Display bot status"""
        print(f"\n{Colors.CYAN}{'─' * 60}{Colors.END}")
        print(f"{Colors.BOLD}Status Update{Colors.END}")
        print(f"Balance: {balance:.2f} {self.quote_asset}")
        print(f"Price: {price:.2f} {self.quote_asset}")
        print(f"Daily P&L: {daily_pnl:+.2f} {self.quote_asset}")
        print(f"Signals Processed: {self.signals_processed}")
        
        if position:
            print(f"{Colors.GREEN}Position: {position['side']} {position['quantity']:.6f} @ {position['entry_price']:.2f}{Colors.END}")
        else:
            print("Position: None")
        
        print(f"{Colors.CYAN}{'─' * 60}{Colors.END}\n")
    
    def shutdown(self):
        """Clean shutdown"""
        print(f"\n{Colors.YELLOW}Shutting down...{Colors.END}")
        self.running = False
        
        # Close any open positions if configured to do so
        # (Optional - comment out if you want positions to stay open)
        # try:
        #     position = self.binance.get_position(self.config.trading_pair)
        #     if position:
        #         print(f"{Colors.YELLOW}Closing open position...{Colors.END}")
        #         self.binance.close_position(self.config.trading_pair)
        # except Exception as e:
        #     print(f"{Colors.RED}Error closing position: {e}{Colors.END}")
        
        self.db.info("Bot shutdown")
        print(f"{Colors.GREEN}Shutdown complete{Colors.END}")


def signal_handler(signum, frame):
    """Handle Ctrl+C"""
    print(f"\n{Colors.YELLOW}Received interrupt signal{Colors.END}")
    if 'bot' in globals():
        bot.shutdown()
    sys.exit(0)


def main():
    """Main entry point"""
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize configuration
    config = Config()
    
    # Get configuration from user
    if not config.get_from_user():
        sys.exit(0)
    
    # Validate configuration
    if not config.validate():
        print(f"{Colors.RED}Invalid configuration{Colors.END}")
        sys.exit(1)
    
    # Initialize database
    db = Database('bot.db')
    db.info("Bot starting up")
    
    # Create and initialize bot
    global bot
    bot = FuturesBot(config, db)
    
    if not bot.initialize():
        print(f"{Colors.RED}Failed to initialize bot{Colors.END}")
        sys.exit(1)
    
    # Start trading
    try:
        bot.trading_loop()
    except KeyboardInterrupt:
        pass
    finally:
        bot.shutdown()


if __name__ == '__main__':
    main()

