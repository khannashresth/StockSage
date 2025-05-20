import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import talib
import logging

class ChartPattern(Enum):
    DOUBLE_BOTTOM = "Double Bottom"
    DOUBLE_TOP = "Double Top"
    HEAD_SHOULDERS = "Head and Shoulders"
    INV_HEAD_SHOULDERS = "Inverse Head and Shoulders"
    TRIANGLE_ASCENDING = "Ascending Triangle"
    TRIANGLE_DESCENDING = "Descending Triangle"
    TRIANGLE_SYMMETRICAL = "Symmetrical Triangle"
    CHANNEL_UP = "Rising Channel"
    CHANNEL_DOWN = "Falling Channel"

@dataclass
class DrawingTool:
    tool_type: str
    points: List[Tuple[pd.Timestamp, float]]
    color: str
    name: str

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_bollinger_bands(data: pd.Series, period: int = 20, num_std: float = 2) -> tuple:
    """Calculate Bollinger Bands"""
    middle_band = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    return upper_band, middle_band, lower_band

def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """Calculate MACD"""
    exp1 = data.ewm(span=fast_period, adjust=False).mean()
    exp2 = data.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple:
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d

class AdvancedChartingSystem:
    def __init__(self, data: pd.DataFrame):
        """Initialize the charting system with proper data conversion."""
        # Ensure data is properly formatted
        self.data = data.copy()
        # Convert price data to numpy arrays for technical indicators
        self.np_close = self.data['Close'].to_numpy()
        self.np_high = self.data['High'].to_numpy()
        self.np_low = self.data['Low'].to_numpy()
        self.np_open = self.data['Open'].to_numpy()
        self.np_volume = self.data['Volume'].to_numpy()
        
        self.drawings = []
        self.recognized_patterns = []
        self.logger = logging.getLogger(__name__)
        
    def add_drawing(self, drawing: DrawingTool):
        """Add a drawing tool to the chart."""
        self.drawings.append(drawing)
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        """Calculate all technical indicators using numpy arrays."""
        try:
            df = self.data.copy()
            
            # Basic indicators using numpy arrays
            df['SMA_20'] = talib.SMA(self.np_close, timeperiod=20)
            df['SMA_50'] = talib.SMA(self.np_close, timeperiod=50)
            df['SMA_200'] = talib.SMA(self.np_close, timeperiod=200)
            
            # MACD
            macd, signal, hist = talib.MACD(self.np_close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['MACD'] = macd
            df['Signal_Line'] = signal
            df['MACD_Histogram'] = hist
            
            # RSI
            df['RSI'] = talib.RSI(self.np_close, timeperiod=14)
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(self.np_close, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['BB_Upper'] = upper
            df['BB_Middle'] = middle
            df['BB_Lower'] = lower
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return self.data.copy()
    
    def recognize_patterns(self) -> List[Dict]:
        """Identify technical patterns in the price data."""
        try:
            patterns = []
            
            # Use pre-converted numpy arrays
            pattern_functions = {
                'Bullish Engulfing': talib.CDLENGULFING,
                'Evening Star': talib.CDLEVENINGSTAR,
                'Morning Star': talib.CDLMORNINGSTAR,
                'Hammer': talib.CDLHAMMER,
                'Shooting Star': talib.CDLSHOOTINGSTAR,
                'Doji': talib.CDLDOJI,
                'Three White Soldiers': talib.CDL3WHITESOLDIERS,
                'Three Black Crows': talib.CDL3BLACKCROWS
            }
            
            for pattern_name, pattern_func in pattern_functions.items():
                try:
                    if pattern_func in [talib.CDLEVENINGSTAR, talib.CDLMORNINGSTAR]:
                        result = pattern_func(self.np_open, self.np_high, self.np_low, self.np_close, penetration=0.3)
                    else:
                        result = pattern_func(self.np_open, self.np_high, self.np_low, self.np_close)
                    
                    last_value = result[-1]
                    
                    if last_value != 0:
                        signal = 'Bullish' if last_value > 0 else 'Bearish'
                        confidence = min(abs(last_value) / 100, 0.95)
                        
                        patterns.append({
                            'pattern': pattern_name,
                            'signal': signal,
                            'confidence': confidence,
                            'description': self._get_pattern_description(pattern_name, signal)
                        })
                except Exception as e:
                    self.logger.warning(f"Error calculating {pattern_name}: {str(e)}")
                    continue
            
            # Trend Analysis using numpy arrays
            try:
                sma20 = talib.SMA(self.np_close, timeperiod=20)
                sma50 = talib.SMA(self.np_close, timeperiod=50)
                
                if sma20[-1] > sma50[-1] and sma20[-2] <= sma50[-2]:
                    patterns.append({
                        'pattern': 'Golden Cross',
                        'signal': 'Bullish',
                        'confidence': 0.85,
                        'description': 'Short-term momentum is turning positive'
                    })
                elif sma20[-1] < sma50[-1] and sma20[-2] >= sma50[-2]:
                    patterns.append({
                        'pattern': 'Death Cross',
                        'signal': 'Bearish',
                        'confidence': 0.85,
                        'description': 'Short-term momentum is turning negative'
                    })
            except Exception as e:
                self.logger.warning(f"Error in trend analysis: {str(e)}")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in pattern recognition: {str(e)}")
            return []
    
    def _get_pattern_description(self, pattern: str, signal: str) -> str:
        """Get detailed description for each pattern."""
        descriptions = {
            'Bullish Engulfing': 'A bullish reversal pattern where a small bearish candle is followed by a large bullish candle that completely engulfs the previous day.',
            'Evening Star': 'A bearish reversal pattern consisting of three candles: a large bullish candle, a small-bodied candle, and a bearish candle.',
            'Morning Star': 'A bullish reversal pattern consisting of three candles: a large bearish candle, a small-bodied candle, and a bullish candle.',
            'Hammer': 'A bullish reversal pattern with a small body and long lower shadow, indicating buying pressure.',
            'Shooting Star': 'A bearish reversal pattern with a small body and long upper shadow, indicating selling pressure.',
            'Doji': 'A candlestick with a small body, indicating market indecision.',
            'Three White Soldiers': 'Three consecutive bullish candles with higher closes, indicating strong buying pressure.',
            'Three Black Crows': 'Three consecutive bearish candles with lower closes, indicating strong selling pressure.'
        }
        
        base_desc = descriptions.get(pattern, '')
        if signal == 'Bullish':
            return f"{base_desc} Suggesting potential upward movement."
        else:
            return f"{base_desc} Suggesting potential downward movement."
    
    def create_advanced_chart(self, indicators: Dict[str, bool] = None) -> go.Figure:
        """Create an advanced chart with selected indicators."""
        if indicators is None:
            indicators = {
                'volume': True,
                'macd': True,
                'rsi': True,
                'bollinger': True,
                'ichimoku': False,
                'fibonacci': False
            }
        
        # Calculate the number of subplots needed
        n_subplots = 1  # Main price chart
        if indicators['volume']: n_subplots += 1
        if indicators['macd']: n_subplots += 1
        if indicators['rsi']: n_subplots += 1
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=n_subplots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5] + [0.15] * (n_subplots - 1),
            subplot_titles=[]
        )
        
        # Add candlestick chart with white lines
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='Price',
                increasing_line_color='#FFFFFF',
                decreasing_line_color='#FFFFFF',
                increasing_fillcolor='rgba(255, 255, 255, 0.5)',
                decreasing_fillcolor='rgba(255, 255, 255, 0.2)',
                line=dict(width=2),
                showlegend=False
            ),
            row=1, col=1
        )
        
        current_row = 2
        
        # Add volume with white bars
        if indicators['volume']:
            fig.add_trace(
                go.Bar(
                    x=self.data.index,
                    y=self.data['Volume'],
                    name='Volume',
                    marker_color='rgba(255, 255, 255, 0.5)',
                    showlegend=False
                ),
                row=current_row, col=1
            )
            current_row += 1
        
        # Add MACD with white lines
        if indicators['macd']:
            macd_data = self.calculate_all_indicators()
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=macd_data['MACD'],
                    name='MACD',
                    line=dict(color='#FFFFFF', width=2)
                ),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=macd_data['Signal_Line'],
                    name='Signal Line',
                    line=dict(color='rgba(255, 255, 255, 0.7)', width=2)
                ),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Bar(
                    x=self.data.index,
                    y=macd_data['MACD_Histogram'],
                    name='MACD Histogram',
                    marker_color='rgba(255, 255, 255, 0.5)',
                    showlegend=False
                ),
                row=current_row, col=1
            )
            current_row += 1
        
        # Add RSI with white line
        if indicators['rsi']:
            rsi_data = self.calculate_all_indicators()
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=rsi_data['RSI'],
                    name='RSI',
                    line=dict(color='#FFFFFF', width=2)
                ),
                row=current_row, col=1
            )
            # Add RSI levels with white lines
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(255, 255, 255, 0.5)", line_width=1, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(255, 255, 255, 0.5)", line_width=1, row=current_row, col=1)
            current_row += 1
        
        # Add Bollinger Bands with white lines
        if indicators['bollinger']:
            bb_data = self.calculate_all_indicators()
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=bb_data['BB_Upper'],
                    name='Upper Band',
                    line=dict(color='rgba(255, 255, 255, 0.7)', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=bb_data['BB_Middle'],
                    name='Middle Band',
                    line=dict(color='#FFFFFF', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=bb_data['BB_Lower'],
                    name='Lower Band',
                    line=dict(color='rgba(255, 255, 255, 0.7)', width=1, dash='dash'),
                    opacity=0.7,
                    fill='tonexty'
                ),
                row=1, col=1
            )
        
        # Update layout for maximum visibility with pure black background
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(
                color="#FFFFFF",
                family="Arial, sans-serif",
                size=12
            ),
            title_font=dict(
                color="#FFFFFF",
                size=24
            ),
            showlegend=True,
            legend=dict(
                bgcolor="black",
                bordercolor="rgba(255, 255, 255, 0.3)",
                borderwidth=1,
                font=dict(color="#FFFFFF")
            ),
            xaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.1)",
                zerolinecolor="rgba(255, 255, 255, 0.1)",
                showgrid=True,
                gridwidth=1,
                title_font=dict(color="#FFFFFF"),
                tickfont=dict(color="#FFFFFF"),
                rangeslider=dict(visible=False)
            ),
            yaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.1)",
                zerolinecolor="rgba(255, 255, 255, 0.1)",
                showgrid=True,
                gridwidth=1,
                title_font=dict(color="#FFFFFF"),
                tickfont=dict(color="#FFFFFF"),
                side="right"
            ),
            margin=dict(t=30, l=10, r=60, b=30)
        )
        
        # Update all subplot axes for better visibility
        for i in range(2, n_subplots + 1):
            fig.update_yaxes(
                gridcolor="rgba(255, 255, 255, 0.1)",
                zerolinecolor="rgba(255, 255, 255, 0.1)",
                showgrid=True,
                gridwidth=1,
                title_font=dict(color="#FFFFFF"),
                tickfont=dict(color="#FFFFFF"),
                row=i, col=1
            )
            fig.update_xaxes(
                gridcolor="rgba(255, 255, 255, 0.1)",
                zerolinecolor="rgba(255, 255, 255, 0.1)",
                showgrid=True,
                gridwidth=1,
                title_font=dict(color="#FFFFFF"),
                tickfont=dict(color="#FFFFFF"),
                row=i, col=1
            )
        
        return fig

def create_market_depth_chart(bids: List[Dict], asks: List[Dict]) -> go.Figure:
    """Create a market depth chart."""
    # TradingView style colors
    colors = {
        'bg': 'rgb(19,23,34)',
        'grid': 'rgba(128,128,128,0.2)',
        'text': 'rgb(255,255,255)',
        'bids': 'rgba(41,98,255,0.4)',
        'asks': 'rgba(255,82,82,0.4)',
        'bid_line': 'rgb(41,98,255)',
        'ask_line': 'rgb(255,82,82)'
    }
    
    # Process data
    bid_prices = [b['price'] for b in bids]
    bid_volumes = np.cumsum([b['volume'] for b in bids])
    ask_prices = [a['price'] for a in asks]
    ask_volumes = np.cumsum([a['volume'] for a in asks])
    
    fig = go.Figure()
    
    # Add bid depth
    fig.add_trace(
        go.Scatter(
            x=bid_prices,
            y=bid_volumes,
            name='Bids',
            fill='tozeroy',
            fillcolor=colors['bids'],
            line=dict(color=colors['bid_line'], width=1)
        )
    )
    
    # Add ask depth
    fig.add_trace(
        go.Scatter(
            x=ask_prices,
            y=ask_volumes,
            name='Asks',
            fill='tozeroy',
            fillcolor=colors['asks'],
            line=dict(color=colors['ask_line'], width=1)
        )
    )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor=colors['bg'],
        paper_bgcolor=colors['bg'],
        font=dict(color=colors['text']),
        title=dict(
            text='Market Depth',
            font=dict(size=20),
            x=0.5,
            y=0.95
        ),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=colors['text']),
            bordercolor='rgba(128,128,128,0.2)',
            x=0.02,
            y=0.98
        ),
        xaxis=dict(
            title='Price',
            gridcolor=colors['grid'],
            zerolinecolor=colors['grid'],
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='Cumulative Volume',
            gridcolor=colors['grid'],
            zerolinecolor=colors['grid'],
            showgrid=True,
            zeroline=False
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified'
    )
    
    return fig

def create_intraday_chart(data: pd.DataFrame) -> go.Figure:
    """Create an intraday chart with 1-minute intervals."""
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    
    fig.update_layout(
        title='Intraday Price Movement',
        yaxis_title='Price',
        template='plotly_dark',
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_comparison_chart(data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """Create a comparison chart for multiple stocks."""
    fig = go.Figure()
    
    for symbol, data in data_dict.items():
        # Normalize prices to percentage change from start
        normalized = (data['Close'] / data['Close'].iloc[0] - 1) * 100
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=normalized,
            name=symbol,
            mode='lines'
        ))
    
    fig.update_layout(
        title='Price Comparison (%)',
        yaxis_title='Change (%)',
        template='plotly_dark',
        showlegend=True
    )
    
    return fig

def find_peaks(x, distance=1):
    """Find peaks in a 1D array."""
    peaks = []
    for i in range(distance, len(x) - distance):
        if all(x[i] > x[i-j] for j in range(1, distance+1)) and \
           all(x[i] > x[i+j] for j in range(1, distance+1)):
            peaks.append(i)
    return np.array(peaks), None 