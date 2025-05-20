import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

class FundamentalAnalysis:
    def __init__(self, symbol: str):
        if not symbol:
            raise ValueError("Symbol cannot be None or empty")
        self.symbol = symbol
        self.stock = yf.Ticker(symbol)
    
    def get_company_profile(self) -> Dict:
        """Get company profile and basic information."""
        try:
            if not self.stock:
                return {'error': 'Invalid stock symbol'}
                
            info = self.stock.info
            if not info:
                return {'error': 'No data available'}
                
            return {
                'Market Cap': info.get('marketCap', 0) / 10000000,  # Convert to Cr
                'PE Ratio': info.get('trailingPE', 0),
                'Beta': info.get('beta', 0),
                '52W High': info.get('fiftyTwoWeekHigh', 0),
                '52W Low': info.get('fiftyTwoWeekLow', 0),
                'Business Summary': info.get('longBusinessSummary', 'Not available')
            }
        except Exception as e:
            return {'error': f'Failed to fetch company profile: {str(e)}'}
    
    def get_detailed_financials(self) -> Dict:
        """Get detailed financial statements."""
        try:
            income_stmt = self.stock.financials
            balance_sheet = self.stock.balance_sheet
            
            return {
                'Income Statement': income_stmt,
                'Balance Sheet': balance_sheet
            }
        except:
            return {'error': 'Failed to fetch financial statements'}
    
    def plot_income_statement(self) -> go.Figure:
        """Plot income statement trends."""
        try:
            income_stmt = self.stock.financials
            
            fig = go.Figure()
            
            for index in income_stmt.index:
                fig.add_trace(
                    go.Scatter(
                        x=income_stmt.columns,
                        y=income_stmt.loc[index],
                        name=index,
                        mode='lines+markers'
                    )
                )
            
            fig.update_layout(
                title='Income Statement Trends',
                xaxis_title='Date',
                yaxis_title='Amount (INR)',
                showlegend=True,
                template="plotly_dark",
                paper_bgcolor="#1a1c23",
                plot_bgcolor="#1a1c23",
                font=dict(color="#e0e0e0"),
                xaxis=dict(
                    gridcolor="#2d3139",
                    zerolinecolor="#2d3139",
                ),
                yaxis=dict(
                    gridcolor="#2d3139",
                    zerolinecolor="#2d3139",
                )
            )
            
            return fig
        except:
            # Return empty figure if data fetch fails
            return go.Figure()
    
    def plot_balance_sheet(self) -> go.Figure:
        """Plot balance sheet composition."""
        try:
            balance_sheet = self.stock.balance_sheet
            
            fig = go.Figure()
            
            for index in balance_sheet.index:
                fig.add_trace(
                    go.Bar(
                        x=balance_sheet.columns,
                        y=balance_sheet.loc[index],
                        name=index
                    )
                )
            
            fig.update_layout(
                title='Balance Sheet Composition',
                xaxis_title='Date',
                yaxis_title='Amount (INR)',
                showlegend=True,
                barmode='stack',
                template="plotly_dark",
                paper_bgcolor="#1a1c23",
                plot_bgcolor="#1a1c23",
                font=dict(color="#e0e0e0"),
                xaxis=dict(
                    gridcolor="#2d3139",
                    zerolinecolor="#2d3139",
                ),
                yaxis=dict(
                    gridcolor="#2d3139",
                    zerolinecolor="#2d3139",
                )
            )
            
            return fig
        except:
            return go.Figure()
    
    def plot_financial_ratios(self) -> go.Figure:
        """Plot key financial ratios."""
        try:
            info = self.stock.info
            
            ratios = {
                'P/E Ratio': info.get('trailingPE', 0),
                'P/B Ratio': info.get('priceToBook', 0),
                'Debt to Equity': info.get('debtToEquity', 0),
                'Return on Equity': info.get('returnOnEquity', 0) * 100,
                'Profit Margin': info.get('profitMargins', 0) * 100
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(ratios.keys()),
                    y=list(ratios.values()),
                    marker_color='#00d09c'
                )
            ])
            
            fig.update_layout(
                title='Key Financial Ratios',
                xaxis_title='Ratio',
                yaxis_title='Value',
                showlegend=False,
                template="plotly_dark",
                paper_bgcolor="#1a1c23",
                plot_bgcolor="#1a1c23",
                font=dict(color="#e0e0e0"),
                xaxis=dict(
                    gridcolor="#2d3139",
                    zerolinecolor="#2d3139",
                ),
                yaxis=dict(
                    gridcolor="#2d3139",
                    zerolinecolor="#2d3139",
                )
            )
            
            return fig
        except:
            return go.Figure()
    
    def get_peer_comparison(self, peer_symbols: List[str]) -> pd.DataFrame:
        """Get comparison with peer companies."""
        try:
            metrics = []
            
            # Get data for main stock
            main_info = self.stock.info
            metrics.append({
                'Symbol': self.symbol,
                'Market Cap (Cr)': main_info.get('marketCap', 0) / 10000000,
                'P/E Ratio': main_info.get('trailingPE', 0),
                'P/B Ratio': main_info.get('priceToBook', 0),
                'ROE (%)': main_info.get('returnOnEquity', 0) * 100
            })
            
            # Get data for peer stocks
            for peer in peer_symbols:
                try:
                    peer_stock = yf.Ticker(peer)
                    peer_info = peer_stock.info
                    metrics.append({
                        'Symbol': peer,
                        'Market Cap (Cr)': peer_info.get('marketCap', 0) / 10000000,
                        'P/E Ratio': peer_info.get('trailingPE', 0),
                        'P/B Ratio': peer_info.get('priceToBook', 0),
                        'ROE (%)': peer_info.get('returnOnEquity', 0) * 100
                    })
                except:
                    continue
            
            return pd.DataFrame(metrics)
        except:
            return pd.DataFrame()
    
    def plot_peer_comparison(self, peer_data: pd.DataFrame) -> go.Figure:
        """Plot peer comparison charts."""
        if peer_data.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Market Cap Comparison',
                'P/E Ratio Comparison',
                'P/B Ratio Comparison',
                'ROE Comparison'
            )
        )
        
        # Market Cap
        fig.add_trace(
            go.Bar(
                x=peer_data['Symbol'],
                y=peer_data['Market Cap (Cr)'],
                name='Market Cap'
            ),
            row=1,
            col=1
        )
        
        # P/E Ratio
        fig.add_trace(
            go.Bar(
                x=peer_data['Symbol'],
                y=peer_data['P/E Ratio'],
                name='P/E Ratio'
            ),
            row=1,
            col=2
        )
        
        # P/B Ratio
        fig.add_trace(
            go.Bar(
                x=peer_data['Symbol'],
                y=peer_data['P/B Ratio'],
                name='P/B Ratio'
            ),
            row=2,
            col=1
        )
        
        # ROE
        fig.add_trace(
            go.Bar(
                x=peer_data['Symbol'],
                y=peer_data['ROE (%)'],
                name='ROE'
            ),
            row=2,
            col=2
        )
        
        fig.update_layout(
            title='Peer Comparison Analysis',
            showlegend=False,
            height=800
        )
        
        return fig
    
    def plot_cash_flow(self) -> go.Figure:
        """Plot cash flow statement trends."""
        try:
            # Get cash flow data
            cash_flow = self.stock.cashflow
            
            if cash_flow.empty:
                return go.Figure()
            
            # Create subplots for different cash flow components
            fig = make_subplots(
                rows=3, 
                cols=1,
                subplot_titles=(
                    'Operating Cash Flow',
                    'Investing Cash Flow',
                    'Financing Cash Flow'
                ),
                vertical_spacing=0.12,
                row_heights=[0.33, 0.33, 0.33]
            )
            
            # Operating Cash Flow
            operating_items = [
                'Operating Cash Flow',
                'Net Income',
                'Change In Working Capital',
                'Depreciation & Amortization'
            ]
            
            for item in operating_items:
                if item in cash_flow.index:
                    fig.add_trace(
                        go.Scatter(
                            x=cash_flow.columns,
                            y=cash_flow.loc[item] / 10000000,  # Convert to Cr
                            name=item,
                            mode='lines+markers'
                        ),
                        row=1,
                        col=1
                    )
            
            # Investing Cash Flow
            investing_items = [
                'Capital Expenditure',
                'Net Investment Purchase And Sale',
                'Net Business Purchase And Sale'
            ]
            
            for item in investing_items:
                if item in cash_flow.index:
                    fig.add_trace(
                        go.Scatter(
                            x=cash_flow.columns,
                            y=cash_flow.loc[item] / 10000000,  # Convert to Cr
                            name=item,
                            mode='lines+markers'
                        ),
                        row=2,
                        col=1
                    )
            
            # Financing Cash Flow
            financing_items = [
                'Free Cash Flow',
                'Dividend Payout',
                'Net Issuance Of Debt',
                'Net Issuance Of Stock'
            ]
            
            for item in financing_items:
                if item in cash_flow.index:
                    fig.add_trace(
                        go.Scatter(
                            x=cash_flow.columns,
                            y=cash_flow.loc[item] / 10000000,  # Convert to Cr
                            name=item,
                            mode='lines+markers'
                        ),
                        row=3,
                        col=1
                    )
            
            # Update layout
            fig.update_layout(
                height=900,
                showlegend=True,
                template="plotly_dark",
                paper_bgcolor="#1a1c23",
                plot_bgcolor="#1a1c23",
                font=dict(color="#e0e0e0"),
                title={
                    'text': 'Cash Flow Analysis',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                legend=dict(
                    yanchor="top",
                    y=1.1,
                    xanchor="left",
                    x=0
                )
            )
            
            # Update axes
            for i in range(1, 4):
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="#2d3139",
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor="#2d3139",
                    row=i,
                    col=1
                )
                fig.update_yaxes(
                    title_text='Amount (₹ Cr)',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="#2d3139",
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor="#2d3139",
                    row=i,
                    col=1
                )
            
            return fig
        except Exception as e:
            print(f"Error plotting cash flow: {str(e)}")
            return go.Figure() 