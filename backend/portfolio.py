import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize
import logging
import json
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    sector: str
    purchase_date: datetime

class Portfolio:
    def __init__(self):
        self.holdings: Dict[str, Position] = {}
        self.cash: float = 0.0
        self.transactions: List[Dict] = []
        self.performance_history: Dict[str, float] = {}
        
        # Setup Redis for caching if available
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=1)
                self.redis_client.ping()  # Test connection
            except:
                self.redis_client = None
                logging.warning("Redis server not available. Using in-memory storage.")
        
        # In-memory cache as fallback
        self.memory_cache = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Initialize ThreadPoolExecutor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def setup_logging(self):
        """Configure logging for portfolio tracking."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('portfolio.log'),
                logging.StreamHandler()
            ]
        )
    
    def add_position(self, symbol: str, quantity: int, price: float, sector: str):
        """Add a new position or update existing position."""
        try:
            if symbol in self.holdings:
                # Update existing position
                current_position = self.holdings[symbol]
                total_quantity = current_position.quantity + quantity
                total_cost = (current_position.quantity * current_position.avg_price) + (quantity * price)
                new_avg_price = total_cost / total_quantity
                
                self.holdings[symbol] = Position(
                    symbol=symbol,
                    quantity=total_quantity,
                    avg_price=new_avg_price,
                    sector=sector,
                    purchase_date=current_position.purchase_date
                )
            else:
                # Add new position
                self.holdings[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    sector=sector,
                    purchase_date=datetime.now()
                )
            
            # Record transaction
            self.transactions.append({
                'type': 'BUY',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update cache
            self._update_cache()
            
            self.logger.info(f"Added position: {symbol}, Quantity: {quantity}, Price: {price}")
        except Exception as e:
            self.logger.error(f"Error adding position: {str(e)}")
            raise
    
    def remove_position(self, symbol: str, quantity: int, price: float):
        """Remove or reduce a position."""
        try:
            if symbol not in self.holdings:
                raise ValueError(f"Position {symbol} not found in portfolio")
            
            position = self.holdings[symbol]
            if quantity > position.quantity:
                raise ValueError(f"Insufficient quantity for {symbol}")
            
            if quantity == position.quantity:
                del self.holdings[symbol]
            else:
                position.quantity -= quantity
            
            # Record transaction
            self.transactions.append({
                'type': 'SELL',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update cache
            self._update_cache()
            
            self.logger.info(f"Removed position: {symbol}, Quantity: {quantity}, Price: {price}")
        except Exception as e:
            self.logger.error(f"Error removing position: {str(e)}")
            raise
    
    def get_positions(self) -> pd.DataFrame:
        """Get current portfolio positions with real-time data."""
        try:
            if not self.holdings:
                return pd.DataFrame()
            
            positions_data = []
            for symbol, position in self.holdings.items():
                # Get real-time price using yfinance
                stock = yf.Ticker(symbol)
                current_price = stock.info.get('regularMarketPrice', 0)
                
                market_value = position.quantity * current_price
                cost_basis = position.quantity * position.avg_price
                unrealized_pl = market_value - cost_basis
                unrealized_pl_pct = (unrealized_pl / cost_basis * 100) if cost_basis != 0 else 0
                
                positions_data.append({
                    'Symbol': symbol,
                    'Quantity': position.quantity,
                    'Avg Price': position.avg_price,
                    'Current Price': current_price,
                    'Market Value': market_value,
                    'Cost Basis': cost_basis,
                    'Unrealized P&L': unrealized_pl,
                    'Unrealized P&L %': unrealized_pl_pct,
                    'Sector': position.sector,
                    'Purchase Date': position.purchase_date
                })
            
            return pd.DataFrame(positions_data)
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return pd.DataFrame()
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive portfolio performance metrics."""
        try:
            positions_df = self.get_positions()
            
            if positions_df.empty:
                return {
                    'Total Value': self.cash,
                    'Total Return %': 0,
                    'Cash': self.cash,
                    'Number of Positions': 0,
                    'Invested Value': 0,
                    'Today P&L': 0,
                    'Today Return %': 0,
                    'Beta': 0,
                    'Sharpe Ratio': 0,
                    'Alpha': 0
                }
            
            total_value = positions_df['Market Value'].sum() + self.cash
            total_cost = positions_df['Cost Basis'].sum()
            total_return = total_value - total_cost - self.cash
            total_return_pct = (total_return / total_cost * 100) if total_cost != 0 else 0
            
            # Calculate daily P&L
            today_pl = positions_df['Unrealized P&L'].sum()
            today_return_pct = (today_pl / total_cost * 100) if total_cost != 0 else 0
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(positions_df)
            
            metrics = {
                'Total Value': total_value,
                'Total Return %': total_return_pct,
                'Cash': self.cash,
                'Number of Positions': len(positions_df),
                'Invested Value': total_cost,
                'Today P&L': today_pl,
                'Today Return %': today_return_pct,
                **risk_metrics
            }
            
            # Update performance history
            self._update_performance_history(total_value)
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def _calculate_risk_metrics(self, positions_df: pd.DataFrame) -> Dict:
        """Calculate portfolio risk metrics."""
        try:
            # Get historical data for all positions
            symbols = positions_df['Symbol'].tolist()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Download data in parallel
            with ThreadPoolExecutor() as executor:
                historical_data = list(executor.map(
                    lambda x: yf.download(x, start=start_date, end=end_date)['Adj Close'],
                    symbols
                ))
            
            # Combine into a single DataFrame
            prices_df = pd.concat(historical_data, axis=1)
            prices_df.columns = symbols
            
            # Calculate daily returns
            returns_df = prices_df.pct_change()
            
            # Calculate portfolio beta
            market_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
            market_returns = market_data.pct_change()
            
            portfolio_returns = returns_df.mean(axis=1)
            beta = np.cov(portfolio_returns, market_returns)[0,1] / np.var(market_returns)
            
            # Calculate Sharpe ratio
            risk_free_rate = 0.03  # Assume 3% risk-free rate
            excess_returns = portfolio_returns - (risk_free_rate / 252)
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / portfolio_returns.std()
            
            # Calculate alpha
            expected_return = risk_free_rate + beta * (market_returns.mean() * 252 - risk_free_rate)
            actual_return = portfolio_returns.mean() * 252
            alpha = actual_return - expected_return
            
            return {
                'Beta': beta,
                'Sharpe Ratio': sharpe_ratio,
                'Alpha': alpha * 100  # Convert to percentage
            }
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return {'Beta': 0, 'Sharpe Ratio': 0, 'Alpha': 0}
    
    def _update_performance_history(self, current_value: float):
        """Update portfolio performance history."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.performance_history[timestamp] = current_value
            
            # Cache performance history
            self._update_cache()
        except Exception as e:
            self.logger.error(f"Error updating performance history: {str(e)}")
    
    def _update_cache(self):
        """Update cache with current portfolio state."""
        try:
            portfolio_state = {
                'holdings': {
                    symbol: {
                        'quantity': pos.quantity,
                        'avg_price': pos.avg_price,
                        'sector': pos.sector,
                        'purchase_date': pos.purchase_date.isoformat()
                    }
                    for symbol, pos in self.holdings.items()
                },
                'cash': self.cash,
                'transactions': self.transactions
            }
            
            if self.redis_client:
                self.redis_client.set('portfolio_state', json.dumps(portfolio_state))
            else:
                self.memory_cache['portfolio_state'] = {
                    'data': portfolio_state,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            self.logger.error(f"Error updating cache: {str(e)}")
    
    def _get_cached_data(self, key: str) -> Optional[Dict]:
        """Retrieve cached data with fallback to in-memory storage."""
        try:
            if self.redis_client:
                data = self.redis_client.get(key)
                return json.loads(data) if data else None
            else:
                cached = self.memory_cache.get(key)
                if cached:
                    # Check if cache is still valid (less than 1 hour old)
                    if (datetime.now() - cached['timestamp']).seconds < 3600:
                        return cached['data']
                    else:
                        del self.memory_cache[key]
                return None
        except Exception as e:
            self.logger.error(f"Cache retrieval error: {str(e)}")
            return None
    
    def plot_portfolio_composition(self) -> go.Figure:
        """Create an interactive portfolio composition visualization."""
        try:
            positions_df = self.get_positions()
            
            if positions_df.empty:
                return go.Figure()
            
            # Sector allocation
            sector_allocation = positions_df.groupby('Sector')['Market Value'].sum()
            
            # Create donut chart
            fig = go.Figure(data=[go.Pie(
                labels=sector_allocation.index,
                values=sector_allocation.values,
                hole=0.5,
                textinfo='label+percent',
                marker=dict(colors=['#00d09c', '#5367ff', '#ffb61b', '#ff4757', '#8e44ad', '#3498db']),
                textfont=dict(size=12)
            )])
            
            fig.update_layout(
                title='Portfolio Sector Allocation',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12, color='#44475b')
            )
            
            return fig
        except Exception as e:
            self.logger.error(f"Error creating portfolio composition plot: {str(e)}")
            return go.Figure()
    
    def plot_performance_history(self) -> go.Figure:
        """Create an interactive performance history visualization."""
        try:
            if not self.performance_history:
                return go.Figure()
            
            dates = list(self.performance_history.keys())
            values = list(self.performance_history.values())
            
            # Calculate daily returns
            returns = np.diff(values) / values[:-1]
            
            # Create subplot with shared x-axis
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Portfolio Value', 'Daily Returns'),
                row_heights=[0.7, 0.3]
            )
            
            # Portfolio value line
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#00d09c', width=2)
                ),
                row=1, col=1
            )
            
            # Daily returns bar chart
            fig.add_trace(
                go.Bar(
                    x=dates[1:],
                    y=returns,
                    name='Daily Returns',
                    marker_color=np.where(returns >= 0, '#00d09c', '#ff4757')
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12, color='#44475b'),
                xaxis2=dict(showgrid=True, gridwidth=1, gridcolor='#f0f0f0'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f0f0f0'),
                yaxis2=dict(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            )
            
            return fig
        except Exception as e:
            self.logger.error(f"Error creating performance history plot: {str(e)}")
            return go.Figure()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass
        if hasattr(self, 'executor'):
            self.executor.shutdown()
    
    def get_risk_metrics(self) -> Dict:
        """Calculate and return portfolio risk metrics."""
        try:
            # Get historical data for portfolio stocks
            portfolio_returns = pd.DataFrame()
            market_returns = None  # Will store ^NSEI (Nifty 50) returns
            
            # Get market returns (using Nifty 50 as benchmark)
            try:
                market_data = yf.download('^NSEI', period='1y')['Adj Close']
                market_returns = market_data.pct_change().dropna()
            except:
                self.logger.warning("Could not fetch market data for risk calculations")
            
            # Calculate portfolio returns
            for symbol in self.holdings:
                try:
                    data = yf.download(symbol, period='1y')['Adj Close']
                    returns = data.pct_change().dropna()
                    portfolio_returns[symbol] = returns
                except:
                    self.logger.warning(f"Could not fetch data for {symbol}")
            
            if portfolio_returns.empty:
                return {}
            
            # Calculate portfolio metrics
            portfolio_weights = [pos.quantity * pos.avg_price / self.get_total_value() for pos in self.holdings.values()]
            portfolio_daily_returns = (portfolio_returns * portfolio_weights).sum(axis=1)
            
            # Calculate risk metrics
            risk_free_rate = 0.05  # Assuming 5% risk-free rate
            excess_returns = portfolio_daily_returns - (risk_free_rate / 252)  # Daily risk-free rate
            
            metrics = {
                'Beta': self._calculate_beta(portfolio_daily_returns, market_returns) if market_returns is not None else 1.0,
                'Sharpe Ratio': self._calculate_sharpe_ratio(portfolio_daily_returns, risk_free_rate),
                'Alpha': self._calculate_alpha(portfolio_daily_returns, market_returns, risk_free_rate) if market_returns is not None else 0.0,
                'Value at Risk (95%)': self._calculate_var(portfolio_daily_returns, confidence_level=0.95)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    def _calculate_beta(self, portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate portfolio beta."""
        covariance = portfolio_returns.cov(market_returns)
        market_variance = market_returns.var()
        return covariance / market_variance if market_variance != 0 else 1.0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - (risk_free_rate / 252)
        if len(excess_returns) > 0:
            return np.sqrt(252) * (excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0)
        return 0.0
    
    def _calculate_alpha(self, portfolio_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Jensen's Alpha."""
        if market_returns is None:
            return 0.0
        
        beta = self._calculate_beta(portfolio_returns, market_returns)
        portfolio_return = portfolio_returns.mean() * 252
        market_return = market_returns.mean() * 252
        return portfolio_return - (risk_free_rate + beta * (market_return - risk_free_rate))
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if len(returns) > 0:
            return -np.percentile(returns, (1 - confidence_level) * 100)

class PortfolioOptimizer:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.risk_free_rate = 0.04  # 4% risk-free rate
        
    def load_data(self):
        """Load and prepare data for optimization."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        try:
            # Download data for all symbols at once
            self.data = yf.download(self.symbols, start=start_date, end=end_date)['Close']
            self.returns = self.data.pct_change().dropna()
            self.mean_returns = self.returns.mean() * 252  # Annualized returns
            self.cov_matrix = self.returns.cov() * 252  # Annualized covariance
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def optimize_portfolio(self, optimization_type: str = 'efficient_frontier') -> Dict:
        """
        Optimize portfolio based on different strategies.
        
        Args:
            optimization_type: One of ['efficient_frontier', 'max_sharpe', 'min_volatility', 'max_sortino']
        """
        if not self.load_data():
            return None
        
        num_assets = len(self.symbols)
        
        if optimization_type == 'max_sharpe':
            # Maximize Sharpe Ratio
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
                {'type': 'ineq', 'fun': lambda x: x},  # weights >= 0
                {'type': 'ineq', 'fun': lambda x: 0.4 - x}  # weights <= 0.4 (diversification)
            ]
            
            result = minimize(
                lambda w: -self._calculate_sharpe_ratio(w),
                x0=np.array([1/num_assets] * num_assets),
                method='SLSQP',
                constraints=constraints
            )
            
            optimal_weights = result.x
            
        elif optimization_type == 'min_volatility':
            # Minimize Volatility
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: x},
                {'type': 'ineq', 'fun': lambda x: 0.4 - x}
            ]
            
            result = minimize(
                lambda w: self._calculate_portfolio_volatility(w),
                x0=np.array([1/num_assets] * num_assets),
                method='SLSQP',
                constraints=constraints
            )
            
            optimal_weights = result.x
            
        elif optimization_type == 'max_sortino':
            # Maximize Sortino Ratio
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: x},
                {'type': 'ineq', 'fun': lambda x: 0.4 - x}
            ]
            
            result = minimize(
                lambda w: -self._calculate_sortino_ratio(w),
                x0=np.array([1/num_assets] * num_assets),
                method='SLSQP',
                constraints=constraints
            )
            
            optimal_weights = result.x
            
        else:  # efficient_frontier
            return self._generate_efficient_frontier()
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(optimal_weights)
        
        return {
            'weights': dict(zip(self.symbols, optimal_weights)),
            'expected_return': portfolio_metrics['return'],
            'volatility': portfolio_metrics['volatility'],
            'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
            'sortino_ratio': portfolio_metrics['sortino_ratio']
        }
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict:
        """Calculate various portfolio metrics."""
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        sortino_ratio = self._calculate_sortino_ratio(weights)
        
        return {
            'return': portfolio_return * 100,  # Convert to percentage
            'volatility': portfolio_volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio
        }
    
    def _calculate_sharpe_ratio(self, weights: np.ndarray) -> float:
        """Calculate portfolio Sharpe ratio."""
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return (portfolio_return - self.risk_free_rate) / portfolio_volatility
    
    def _calculate_sortino_ratio(self, weights: np.ndarray) -> float:
        """Calculate portfolio Sortino ratio."""
        portfolio_return = np.sum(self.mean_returns * weights)
        
        # Calculate downside returns
        portfolio_returns = np.sum(self.returns.mul(weights, axis=1), axis=1)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = np.sqrt(252) * downside_returns.std()
        
        return (portfolio_return - self.risk_free_rate) / downside_std if downside_std != 0 else 0
    
    def _calculate_portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility."""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * 100
    
    def _generate_efficient_frontier(self) -> Dict:
        """Generate efficient frontier points."""
        num_portfolios = 1000
        returns_array = np.zeros(num_portfolios)
        volatility_array = np.zeros(num_portfolios)
        sharpe_array = np.zeros(num_portfolios)
        weight_array = np.zeros((num_portfolios, len(self.symbols)))
        
        for i in range(num_portfolios):
            weights = np.random.random(len(self.symbols))
            weights = weights / np.sum(weights)
            weight_array[i,:] = weights
            
            portfolio_metrics = self._calculate_portfolio_metrics(weights)
            returns_array[i] = portfolio_metrics['return']
            volatility_array[i] = portfolio_metrics['volatility']
            sharpe_array[i] = portfolio_metrics['sharpe_ratio']
        
        # Find optimal portfolios
        max_sharpe_idx = sharpe_array.argmax()
        min_vol_idx = volatility_array.argmin()
        
        return {
            'efficient_frontier': {
                'returns': returns_array.tolist(),
                'volatility': volatility_array.tolist(),
                'sharpe_ratios': sharpe_array.tolist()
            },
            'max_sharpe_portfolio': {
                'weights': dict(zip(self.symbols, weight_array[max_sharpe_idx])),
                'return': returns_array[max_sharpe_idx],
                'volatility': volatility_array[max_sharpe_idx],
                'sharpe_ratio': sharpe_array[max_sharpe_idx]
            },
            'min_volatility_portfolio': {
                'weights': dict(zip(self.symbols, weight_array[min_vol_idx])),
                'return': returns_array[min_vol_idx],
                'volatility': volatility_array[min_vol_idx],
                'sharpe_ratio': sharpe_array[min_vol_idx]
            }
        }
    
    def plot_efficient_frontier(self) -> go.Figure:
        """Plot the efficient frontier with optimal portfolios."""
        efficient_frontier = self._generate_efficient_frontier()
        
        fig = go.Figure()
        
        # Plot random portfolios
        fig.add_trace(go.Scatter(
            x=efficient_frontier['efficient_frontier']['volatility'],
            y=efficient_frontier['efficient_frontier']['returns'],
            mode='markers',
            marker=dict(
                size=5,
                color=efficient_frontier['efficient_frontier']['sharpe_ratios'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe Ratio')
            ),
            name='Portfolios'
        ))
        
        # Plot maximum Sharpe ratio portfolio
        max_sharpe = efficient_frontier['max_sharpe_portfolio']
        fig.add_trace(go.Scatter(
            x=[max_sharpe['volatility']],
            y=[max_sharpe['return']],
            mode='markers',
            marker=dict(
                size=15,
                symbol='star',
                color='#00d09c'
            ),
            name='Maximum Sharpe Ratio'
        ))
        
        # Plot minimum volatility portfolio
        min_vol = efficient_frontier['min_volatility_portfolio']
        fig.add_trace(go.Scatter(
            x=[min_vol['volatility']],
            y=[min_vol['return']],
            mode='markers',
            marker=dict(
                size=15,
                symbol='star',
                color='#ff4757'
            ),
            name='Minimum Volatility'
        ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Volatility (%)',
            yaxis_title='Expected Return (%)',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                gridcolor='#f0f0f0',
                zerolinecolor='#f0f0f0'
            ),
            yaxis=dict(
                gridcolor='#f0f0f0',
                zerolinecolor='#f0f0f0'
            )
        )
        
        return fig 