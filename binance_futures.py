"""
Binance Futures API Wrapper
Clean REST + WebSocket integration with proper error handling
"""

import time
from decimal import Decimal
from typing import Dict, List, Optional, Callable
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd


class BinanceFutures:
    """Binance Futures API wrapper"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Initialize client
        if testnet:
            self.client = Client(api_key, api_secret, testnet=True)
            self.base_url = 'https://testnet.binancefuture.com'
        else:
            self.client = Client(api_key, api_secret)
            self.base_url = 'https://fapi.binance.com'
        
        # Cache for symbol info
        self._symbol_info_cache = {}
    
    # === MARKET INFO ===
    
    def get_funding_rate(self, symbol: str) -> Dict:
        """
        Get current funding rate for symbol
        
        Returns:
            Dict with:
                - funding_rate: Current funding rate (e.g., 0.0001 = 0.01%)
                - next_funding_time: Next funding time (timestamp)
                - mark_price: Current mark price
        """
        try:
            # Get premium index (contains funding rate)
            premium = self.client.futures_mark_price(symbol=symbol)
            
            funding_rate = float(premium['lastFundingRate'])
            next_funding_time = int(premium['nextFundingTime'])
            mark_price = float(premium['markPrice'])
            
            return {
                'funding_rate': funding_rate,
                'funding_rate_pct': funding_rate * 100,  # As percentage
                'next_funding_time': next_funding_time,
                'mark_price': mark_price
            }
        except BinanceAPIException as e:
            raise Exception(f"Failed to get funding rate: {e}")
    
    def get_funding_rate_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get historical funding rates
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            limit: Number of historical rates to retrieve
            
        Returns:
            List of funding rate records
        """
        try:
            history = self.client.futures_funding_rate(symbol=symbol, limit=limit)
            
            results = []
            for record in history:
                results.append({
                    'funding_rate': float(record['fundingRate']),
                    'funding_rate_pct': float(record['fundingRate']) * 100,
                    'funding_time': int(record['fundingTime'])
                })
            
            return results
        except BinanceAPIException as e:
            raise Exception(f"Failed to get funding rate history: {e}")
    
    # === ACCOUNT INFO ===
    
    def get_balance(self, quote_asset: str = 'USDT') -> float:
        """
        Get balance for specified quote asset
        
        Args:
            quote_asset: Quote currency (USDT, USDC, BUSD, etc.)
        """
        try:
            account = self.client.futures_account()
            for asset in account['assets']:
                if asset['asset'] == quote_asset:
                    return float(asset['availableBalance'])
            return 0.0
        except BinanceAPIException as e:
            raise Exception(f"Failed to get balance: {e}")
    
    def get_quote_asset(self, symbol: str) -> str:
        """
        Extract quote asset from trading pair
        
        Examples:
            BTCUSDT → USDT
            ETHUSDC → USDC
            BNBBUSD → BUSD
        """
        # Common quote assets (check from longest to shortest)
        quote_assets = ['USDT', 'USDC', 'BUSD', 'USD', 'BTC', 'ETH', 'BNB']
        
        for quote in quote_assets:
            if symbol.endswith(quote):
                return quote
        
        # Default to USDT if not found
        return 'USDT'
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for symbol"""
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            for pos in positions:
                if float(pos['positionAmt']) != 0:
                    return {
                        'symbol': pos['symbol'],
                        'side': 'LONG' if float(pos['positionAmt']) > 0 else 'SHORT',
                        'quantity': abs(float(pos['positionAmt'])),
                        'entry_price': float(pos['entryPrice']),
                        'unrealized_pnl': float(pos['unRealizedProfit']),
                        'leverage': int(pos.get('leverage', 1))  # Default to 1 if not present
                    }
            return None
        except BinanceAPIException as e:
            raise Exception(f"Failed to get position: {e}")
    
    # === SYMBOL INFO ===
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol trading rules"""
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]
        
        try:
            exchange_info = self.client.futures_exchange_info()
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    # Extract important filters
                    info = {
                        'symbol': symbol,
                        'price_precision': s['pricePrecision'],
                        'quantity_precision': s['quantityPrecision'],
                        'min_qty': 0,
                        'step_size': 0,
                        'tick_size': 0
                    }
                    
                    for f in s['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            info['min_qty'] = float(f['minQty'])
                            info['step_size'] = float(f['stepSize'])
                        elif f['filterType'] == 'PRICE_FILTER':
                            info['tick_size'] = float(f['tickSize'])
                    
                    self._symbol_info_cache[symbol] = info
                    return info
            
            raise Exception(f"Symbol {symbol} not found")
        except BinanceAPIException as e:
            raise Exception(f"Failed to get symbol info: {e}")
    
    def round_quantity(self, symbol: str, quantity: float) -> str:
        """Round quantity to symbol precision"""
        info = self.get_symbol_info(symbol)
        step_size = Decimal(str(info['step_size']))
        quantity_dec = Decimal(str(quantity))
        
        # Round to step size
        rounded = (quantity_dec // step_size) * step_size
        
        # Format to avoid scientific notation
        precision = info['quantity_precision']
        return f"{rounded:.{precision}f}"
    
    def round_price(self, symbol: str, price: float) -> str:
        """Round price to symbol precision"""
        info = self.get_symbol_info(symbol)
        tick_size = Decimal(str(info['tick_size']))
        price_dec = Decimal(str(price))
        
        # Round to tick size
        rounded = (price_dec // tick_size) * tick_size
        
        # Format to avoid scientific notation
        precision = info['price_precision']
        return f"{rounded:.{precision}f}"
    
    # === LEVERAGE & MARGIN ===
    
    def set_leverage(self, symbol: str, leverage: int):
        """Set leverage for symbol"""
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
        except BinanceAPIException as e:
            # Ignore if leverage already set
            if 'No need to change leverage' not in str(e):
                raise Exception(f"Failed to set leverage: {e}")
    
    def set_margin_type(self, symbol: str, margin_type: str):
        """Set margin type (ISOLATED or CROSSED)"""
        try:
            self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
        except BinanceAPIException as e:
            # Ignore if margin type already set
            if 'No need to change margin type' not in str(e):
                raise Exception(f"Failed to set margin type: {e}")
    
    def set_position_mode(self, dual_side: bool = False):
        """
        Set position mode
        
        Args:
            dual_side: False = One-Way Mode (default), True = Hedge Mode
        """
        try:
            self.client.futures_change_position_mode(dualSidePosition=dual_side)
        except BinanceAPIException as e:
            # Ignore if position mode already set
            if 'No need to change position side' in str(e):
                pass
            else:
                # This error is OK - means it's already in the desired mode
                pass
    
    # === ORDERS ===
    
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Place market order"""
        try:
            qty_str = self.round_quantity(symbol, quantity)
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,  # BUY or SELL
                type='MARKET',
                quantity=qty_str
            )
            
            # Market orders may not have avgPrice immediately
            # Get the actual fill price from order info or current price
            avg_price = float(order.get('avgPrice', 0))
            
            if avg_price == 0:
                # If avgPrice is 0, query the order to get actual fill price
                try:
                    order_info = self.client.futures_get_order(
                        symbol=symbol,
                        orderId=order['orderId']
                    )
                    avg_price = float(order_info.get('avgPrice', 0))
                except:
                    pass
                
                # If still 0, use current market price as fallback
                if avg_price == 0:
                    avg_price = self.get_current_price(symbol)
            
            return {
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'price': avg_price,
                'quantity': float(order['executedQty']),
                'status': order['status']
            }
        except BinanceAPIException as e:
            raise Exception(f"Market order failed: {e}")
    
    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """Place limit order"""
        try:
            qty_str = self.round_quantity(symbol, quantity)
            price_str = self.round_price(symbol, price)
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=qty_str,
                price=price_str
            )
            
            return {
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'price': float(order['price']),
                'quantity': float(order['origQty']),
                'status': order['status']
            }
        except BinanceAPIException as e:
            raise Exception(f"Limit order failed: {e}")
    
    def place_stop_loss(self, symbol: str, side: str, quantity: float, stop_price: float) -> Dict:
        """Place stop loss order"""
        try:
            qty_str = self.round_quantity(symbol, quantity)
            stop_price_str = self.round_price(symbol, stop_price)
            
            # Stop side is opposite of position side
            # side can be 'BUY'/'SELL' or 'LONG'/'SHORT'
            stop_side = 'SELL' if side in ['LONG', 'BUY'] else 'BUY'
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=stop_side,
                type='STOP_MARKET',
                stopPrice=stop_price_str,
                quantity=qty_str,
                closePosition=True
            )
            
            return {
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'stop_price': float(order['stopPrice']),
                'quantity': float(order['origQty']),
                'status': order['status']
            }
        except BinanceAPIException as e:
            raise Exception(f"Stop loss order failed: {e}")
    
    def place_take_profit(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """Place take profit order"""
        try:
            qty_str = self.round_quantity(symbol, quantity)
            price_str = self.round_price(symbol, price)
            
            # TP side is opposite of position side
            # side can be 'BUY'/'SELL' or 'LONG'/'SHORT'
            tp_side = 'SELL' if side in ['LONG', 'BUY'] else 'BUY'
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side=tp_side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=price_str,
                quantity=qty_str,
                closePosition=True
            )
            
            return {
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'price': float(order['stopPrice']),
                'quantity': float(order['origQty']),
                'status': order['status']
            }
        except BinanceAPIException as e:
            raise Exception(f"Take profit order failed: {e}")
    
    def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for symbol"""
        try:
            self.client.futures_cancel_all_open_orders(symbol=symbol)
        except BinanceAPIException as e:
            raise Exception(f"Failed to cancel orders: {e}")
    
    # === MARKET DATA ===
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            raise Exception(f"Failed to get price: {e}")
    
    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """Get historical klines/candles"""
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except BinanceAPIException as e:
            raise Exception(f"Failed to get klines: {e}")
    
    # === POSITION MANAGEMENT ===
    
    def close_position(self, symbol: str) -> Optional[Dict]:
        """Close entire position"""
        position = self.get_position(symbol)
        if not position:
            return None
        
        # Cancel all open orders first
        self.cancel_all_orders(symbol)
        
        # Close position with market order
        close_side = 'SELL' if position['side'] == 'LONG' else 'BUY'
        return self.place_market_order(symbol, close_side, position['quantity'])
    
    # === TRADE HISTORY ===
    
    def get_account_trades(self, symbol: str = None, limit: int = 500, 
                          start_time: int = None, end_time: int = None) -> List[Dict]:
        """
        Get account trade history
        
        Args:
            symbol: Trading pair (optional, get all if None)
            limit: Max trades to return (default 500, max 1000)
            start_time: Start timestamp in ms (optional)
            end_time: End timestamp in ms (optional)
        
        Returns:
            List of trade records with P&L information
        """
        try:
            params = {'limit': min(limit, 1000)}
            
            if symbol:
                params['symbol'] = symbol
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            trades = self.client.futures_account_trades(**params)
            
            results = []
            for trade in trades:
                results.append({
                    'symbol': trade['symbol'],
                    'id': trade['id'],
                    'order_id': trade['orderId'],
                    'side': trade['side'],  # BUY or SELL
                    'price': float(trade['price']),
                    'quantity': float(trade['qty']),
                    'quote_qty': float(trade['quoteQty']),
                    'commission': float(trade['commission']),
                    'commission_asset': trade['commissionAsset'],
                    'time': int(trade['time']),
                    'position_side': trade['positionSide'],
                    'realized_pnl': float(trade['realizedPnl']),
                    'is_maker': trade['maker']
                })
            
            return results
        except BinanceAPIException as e:
            raise Exception(f"Failed to get account trades: {e}")
    
    def get_income_history(self, symbol: str = None, income_type: str = None,
                          limit: int = 100, start_time: int = None, 
                          end_time: int = None) -> List[Dict]:
        """
        Get account income history (P&L, funding fees, commissions, etc.)
        
        Args:
            symbol: Trading pair (optional)
            income_type: Type of income (REALIZED_PNL, FUNDING_FEE, COMMISSION, etc.)
            limit: Max records to return (default 100, max 1000)
            start_time: Start timestamp in ms (optional)
            end_time: End timestamp in ms (optional)
        
        Returns:
            List of income records
        """
        try:
            params = {'limit': min(limit, 1000)}
            
            if symbol:
                params['symbol'] = symbol
            if income_type:
                params['incomeType'] = income_type
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            incomes = self.client.futures_income_history(**params)
            
            results = []
            for income in incomes:
                results.append({
                    'symbol': income['symbol'],
                    'income_type': income['incomeType'],
                    'income': float(income['income']),
                    'asset': income['asset'],
                    'info': income.get('info', ''),
                    'time': int(income['time']),
                    'tran_id': income['tranId'],
                    'trade_id': income.get('tradeId', '')
                })
            
            return results
        except BinanceAPIException as e:
            raise Exception(f"Failed to get income history: {e}")

