"""
Risk Management Module
Position sizing, safety checks, and risk validation
"""

from typing import Dict, Optional


class RiskManager:
    """Handles position sizing and risk checks"""
    
    def __init__(self, config):
        self.config = config
        self.max_position_pct = config.max_position_pct
        self.stop_loss_pct = config.stop_loss_pct
        self.take_profit_pct = config.take_profit_pct
        self.max_daily_loss_pct = config.max_daily_loss_pct
        self.leverage = config.leverage
    
    def calculate_position_size(self, balance: float, price: float) -> float:
        """
        Calculate position size based on balance and leverage
        
        Returns quantity in base asset
        """
        # Available for position (% of balance)
        position_value = balance * (self.max_position_pct / 100.0)
        
        # With leverage
        position_value_leveraged = position_value * self.leverage
        
        # Quantity in base asset
        quantity = position_value_leveraged / price
        
        return quantity
    
    def calculate_stop_loss_price(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price"""
        if side == 'BUY' or side == 'LONG':
            # Long position: stop below entry
            return entry_price * (1 - self.stop_loss_pct / 100.0)
        else:
            # Short position: stop above entry
            return entry_price * (1 + self.stop_loss_pct / 100.0)
    
    def calculate_take_profit_price(self, entry_price: float, side: str) -> float:
        """Calculate take profit price"""
        if side == 'BUY' or side == 'LONG':
            # Long position: profit above entry
            return entry_price * (1 + self.take_profit_pct / 100.0)
        else:
            # Short position: profit below entry
            return entry_price * (1 - self.take_profit_pct / 100.0)
    
    def calculate_liquidation_price(self, entry_price: float, side: str, leverage: int) -> float:
        """
        Estimate liquidation price
        Simplified calculation for ISOLATED margin
        """
        # Maintenance margin rate (approximate)
        mmr = 0.004  # 0.4% for most pairs
        
        if side == 'BUY' or side == 'LONG':
            # Long liquidation
            liq_price = entry_price * (1 - (1 / leverage) + mmr)
        else:
            # Short liquidation
            liq_price = entry_price * (1 + (1 / leverage) - mmr)
        
        return liq_price
    
    def check_daily_loss_limit(self, daily_pnl: float, balance: float) -> bool:
        """
        Check if daily loss limit exceeded
        Returns True if safe to trade, False if limit exceeded
        """
        max_daily_loss = balance * (self.max_daily_loss_pct / 100.0)
        
        if daily_pnl < -max_daily_loss:
            return False
        
        return True
    
    def check_liquidation_risk(self, current_price: float, entry_price: float, 
                              side: str, leverage: int) -> Dict:
        """
        Check liquidation risk
        Returns risk assessment
        """
        liq_price = self.calculate_liquidation_price(entry_price, side, leverage)
        
        if side == 'BUY' or side == 'LONG':
            distance_pct = ((current_price - liq_price) / liq_price) * 100
        else:
            distance_pct = ((liq_price - current_price) / current_price) * 100
        
        # Risk levels
        if distance_pct < 5:
            risk_level = 'CRITICAL'
        elif distance_pct < 15:
            risk_level = 'HIGH'
        elif distance_pct < 30:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'liquidation_price': liq_price,
            'distance_pct': distance_pct,
            'risk_level': risk_level
        }
    
    def validate_trade(self, balance: float, current_position: Optional[Dict], 
                      daily_pnl: float) -> tuple[bool, str]:
        """
        Validate if trade should be allowed
        Returns (allowed: bool, reason: str)
        """
        # Check daily loss limit
        if not self.check_daily_loss_limit(daily_pnl, balance):
            return False, f"Daily loss limit exceeded ({daily_pnl:.2f} USDT)"
        
        # Check if already in position
        if current_position:
            return False, f"Already in {current_position['side']} position"
        
        # Check minimum balance
        if balance < 10:
            return False, f"Insufficient balance ({balance:.2f} USDT)"
        
        return True, "OK"
    
    def calculate_pnl(self, entry_price: float, exit_price: float, 
                     quantity: float, side: str, leverage: int) -> float:
        """Calculate profit/loss"""
        if side == 'BUY' or side == 'LONG':
            # Long position
            pnl_per_unit = exit_price - entry_price
        else:
            # Short position
            pnl_per_unit = entry_price - exit_price
        
        pnl = pnl_per_unit * quantity
        return pnl
    
    def calculate_trailing_stop_price(self, highest_price: float, side: str) -> float:
        """
        Calculate trailing stop price based on highest/lowest price
        
        Args:
            highest_price: Highest price for LONG, lowest price for SHORT
            side: Position side (BUY/SELL/LONG/SHORT)
        
        Returns:
            Trailing stop price
        """
        # Use stop_loss_pct as trailing distance
        trailing_pct = self.stop_loss_pct
        
        if side in ['BUY', 'LONG']:
            # LONG: stop below highest price
            return highest_price * (1 - trailing_pct / 100.0)
        else:
            # SHORT: stop above lowest price
            return highest_price * (1 + trailing_pct / 100.0)
    
    def should_close_position_trailing(self, position: Dict, current_price: float, 
                                      highest_price: float) -> tuple[bool, str]:
        """
        Check if position should be closed using trailing stop
        
        Args:
            position: Position dict
            current_price: Current market price
            highest_price: Highest price for LONG, lowest for SHORT
        
        Returns:
            (should_close: bool, reason: str)
        """
        entry_price = position['entry_price']
        side = position['side']
        
        # Calculate current P&L %
        if side in ['LONG', 'BUY']:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Calculate trailing stop price
        trailing_stop = self.calculate_trailing_stop_price(highest_price, side)
        
        # Check trailing stop trigger
        if side in ['LONG', 'BUY']:
            # LONG: close if price drops below trailing stop
            if current_price <= trailing_stop:
                return True, f"Trailing stop hit at {trailing_stop:.2f} (from high {highest_price:.2f})"
        else:
            # SHORT: close if price rises above trailing stop
            if current_price >= trailing_stop:
                return True, f"Trailing stop hit at {trailing_stop:.2f} (from low {highest_price:.2f})"
        
        # Check take profit (optional - keep as max target)
        if pnl_pct >= self.take_profit_pct:
            return True, f"Take profit target hit ({pnl_pct:.2f}%)"
        
        return False, ""
    
    def should_close_position(self, position: Dict, current_price: float) -> tuple[bool, str]:
        """
        Check if position should be closed (old method - kept for compatibility)
        Returns (should_close: bool, reason: str)
        """
        entry_price = position['entry_price']
        side = position['side']
        
        # Calculate current P&L %
        if side == 'LONG':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Check stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, f"Stop loss hit ({pnl_pct:.2f}%)"
        
        # Check take profit
        if pnl_pct >= self.take_profit_pct:
            return True, f"Take profit hit ({pnl_pct:.2f}%)"
        
        return False, ""

