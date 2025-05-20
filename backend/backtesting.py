import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None

class TradingStrategy:
    def __init__(self, name: str):
        self.name = name
        self.indicators: Dict[str, Callable] = {}
        self.entry_rules: List[Callable] = []
        self.exit_rules: List[Callable] = []
        self.risk_rules: List[Callable] = []
    
    def add_indicator(self, name: str, func: Callable):
        """Add a technical indicator calculation function."""
        self.indicators[name] = func
    
    def add_entry_rule(self, rule: Callable):
        """Add an entry rule function."""
        self.entry_rules.append(rule)
    
    def add_exit_rule(self, rule: Callable):
        """Add an exit rule function."""
        self.exit_rules.append(rule)
    
    def add_risk_rule(self, rule: Callable):
        """Add a risk management rule function."""
        self.risk_rules.append(rule)
    
    def should_enter(self, data: pd.DataFrame, current_idx: int) -> bool:
        """Check if all entry rules are satisfied."""
        return all(rule(data, current_idx) for rule in self.entry_rules)
    
    def should_exit(self, data: pd.DataFrame, current_idx: int) -> bool:
        """Check if any exit rule is satisfied."""
        return any(rule(data, current_idx) for rule in self.exit_rules)
    
    def check_risk(self, data: pd.DataFrame, current_idx: int) -> bool:
        """Check if all risk rules are satisfied."""
        return all(rule(data, current_idx) for rule in self.risk_rules)

class Backtester:
    def __init__(self, data: pd.DataFrame, strategy: TradingStrategy, 
                 initial_capital: float = 100000.0):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: List[Position] = []
        self.orders: List[Order] = []
        self.equity_curve = []
        
    def run(self) -> Dict:
        """Run the backtest."""
        # Calculate all indicators
        for name, func in self.strategy.indicators.items():
            self.data[name] = func(self.data)
        
        # Simulate trading
        for i in range(len(self.data)):
            # Update equity curve
            self.equity_curve.append(self._calculate_equity(i))
            
            # Check for exit signals
            self._process_exits(i)
            
            # Check for entry signals
            if self.strategy.should_enter(self.data, i) and self.strategy.check_risk(self.data, i):
                self._process_entry(i)
        
        # Close any remaining positions
        self._close_all_positions(len(self.data) - 1)
        
        # Calculate performance metrics
        return self._calculate_performance()
    
    def _process_entry(self, idx: int):
        """Process entry signal."""
        price = self.data.iloc[idx]['Close']
        position_size = self._calculate_position_size(price)
        
        if position_size > 0:
            order = Order(
                symbol=self.data.index.name or "UNKNOWN",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=position_size,
                price=price,
                timestamp=self.data.index[idx]
            )
            self.orders.append(order)
            
            position = Position(
                symbol=order.symbol,
                quantity=order.quantity,
                entry_price=order.price,
                entry_time=order.timestamp
            )
            self.positions.append(position)
            
            self.current_capital -= position_size * price
    
    def _process_exits(self, idx: int):
        """Process exit signals."""
        for position in self.positions:
            if position.exit_time is None and self.strategy.should_exit(self.data, idx):
                price = self.data.iloc[idx]['Close']
                
                order = Order(
                    symbol=position.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity,
                    price=price,
                    timestamp=self.data.index[idx]
                )
                self.orders.append(order)
                
                position.exit_price = price
                position.exit_time = order.timestamp
                position.pnl = (position.exit_price - position.entry_price) * position.quantity
                
                self.current_capital += position.quantity * price
    
    def _close_all_positions(self, idx: int):
        """Close all open positions."""
        price = self.data.iloc[idx]['Close']
        
        for position in self.positions:
            if position.exit_time is None:
                order = Order(
                    symbol=position.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity,
                    price=price,
                    timestamp=self.data.index[idx]
                )
                self.orders.append(order)
                
                position.exit_price = price
                position.exit_time = order.timestamp
                position.pnl = (position.exit_price - position.entry_price) * position.quantity
                
                self.current_capital += position.quantity * price
    
    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk management rules."""
        risk_per_trade = 0.02  # 2% risk per trade
        available_capital = self.current_capital * risk_per_trade
        return available_capital / price
    
    def _calculate_equity(self, idx: int) -> float:
        """Calculate current equity."""
        equity = self.current_capital
        price = self.data.iloc[idx]['Close']
        
        for position in self.positions:
            if position.exit_time is None:
                equity += position.quantity * price
        
        return equity
    
    def _calculate_performance(self) -> Dict:
        """Calculate performance metrics."""
        equity_curve = pd.Series(self.equity_curve, index=self.data.index)
        returns = equity_curve.pct_change().dropna()
        
        total_trades = len(self.orders) // 2
        winning_trades = len([p for p in self.positions if p.pnl and p.pnl > 0])
        losing_trades = len([p for p in self.positions if p.pnl and p.pnl <= 0])
        
        metrics = {
            'Initial Capital': self.initial_capital,
            'Final Capital': self.equity_curve[-1],
            'Total Return': (self.equity_curve[-1] / self.initial_capital - 1) * 100,
            'Annual Return': ((self.equity_curve[-1] / self.initial_capital) ** (252 / len(self.data)) - 1) * 100,
            'Sharpe Ratio': np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0,
            'Max Drawdown': self._calculate_max_drawdown(equity_curve) * 100,
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'Average Win': np.mean([p.pnl for p in self.positions if p.pnl and p.pnl > 0]) if winning_trades > 0 else 0,
            'Average Loss': np.mean([p.pnl for p in self.positions if p.pnl and p.pnl <= 0]) if losing_trades > 0 else 0,
            'Profit Factor': self._calculate_profit_factor(),
            'Recovery Factor': self._calculate_recovery_factor(equity_curve)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1
        return abs(drawdowns.min())
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor."""
        gross_profit = sum([p.pnl for p in self.positions if p.pnl and p.pnl > 0])
        gross_loss = abs(sum([p.pnl for p in self.positions if p.pnl and p.pnl <= 0]))
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    def _calculate_recovery_factor(self, equity_curve: pd.Series) -> float:
        """Calculate recovery factor."""
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        total_return = (equity_curve[-1] / equity_curve[0] - 1)
        return total_return / max_drawdown if max_drawdown != 0 else float('inf')
    
    def plot_results(self) -> go.Figure:
        """Create an interactive plot of backtest results."""
        fig = make_subplots(
            rows=2, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price and Trades', 'Equity Curve'),
            row_heights=[0.7, 0.3]
        )
        
        # Plot price
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='Price'
            ),
            row=1,
            col=1
        )
        
        # Plot trades
        for position in self.positions:
            # Entry point
            fig.add_trace(
                go.Scatter(
                    x=[position.entry_time],
                    y=[position.entry_price],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='green'
                    ),
                    name='Entry'
                ),
                row=1,
                col=1
            )
            
            # Exit point
            if position.exit_time and position.exit_price:
                fig.add_trace(
                    go.Scatter(
                        x=[position.exit_time],
                        y=[position.exit_price],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='red'
                        ),
                        name='Exit'
                    ),
                    row=1,
                    col=1
                )
        
        # Plot equity curve
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.equity_curve,
                name='Equity',
                line=dict(color='blue')
            ),
            row=2,
            col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Backtest Results',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        return fig

def create_strategy(name: str, indicators: Dict[str, Callable],
                   entry_rules: List[Callable], exit_rules: List[Callable],
                   risk_rules: List[Callable]) -> TradingStrategy:
    """Create a trading strategy with the specified rules."""
    strategy = TradingStrategy(name)
    
    for indicator_name, indicator_func in indicators.items():
        strategy.add_indicator(indicator_name, indicator_func)
    
    for entry_rule in entry_rules:
        strategy.add_entry_rule(entry_rule)
    
    for exit_rule in exit_rules:
        strategy.add_exit_rule(exit_rule)
    
    for risk_rule in risk_rules:
        strategy.add_risk_rule(risk_rule)
    
    return strategy 