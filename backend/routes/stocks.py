from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

router = APIRouter()

class StockData(BaseModel):
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime

@router.get("/search/{query}")
async def search_stocks(query: str) -> List[Dict[str, str]]:
    """Search for stocks by symbol or name."""
    try:
        from nse_stocks import NSE_STOCKS
        filtered_stocks = {
            k: v for k, v in NSE_STOCKS.items() 
            if query.upper() in k.upper() or query.upper() in v.upper()
        }
        return [{"symbol": k, "name": v} for k, v in filtered_stocks.items()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}/price")
async def get_stock_price(
    symbol: str,
    interval: Optional[str] = Query("1d", enum=["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"])
) -> StockData:
    """Get real-time stock price data."""
    try:
        from realtime_data import LiveMarketData
        live_data = LiveMarketData()
        data = live_data.get_real_time_price(symbol)
        return StockData(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}/depth")
async def get_market_depth(symbol: str) -> Dict[str, Any]:
    """Get market depth data for a stock."""
    try:
        from realtime_data import LiveMarketData
        live_data = LiveMarketData()
        depth_data = live_data.get_market_depth(symbol)
        if depth_data.get('error'):
            raise HTTPException(status_code=404, detail="Market depth data not available")
        return depth_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}/chart")
async def get_stock_chart(
    symbol: str,
    start_date: datetime,
    end_date: Optional[datetime] = None,
    indicators: Optional[List[str]] = Query(None)
) -> Dict[str, Any]:
    """Get stock chart data with technical indicators."""
    try:
        import yfinance as yf
        from advanced_charts import AdvancedChartingSystem
        
        # Get stock data
        data = yf.download(symbol, start=start_date, end=end_date or datetime.now())
        if data.empty:
            raise HTTPException(status_code=404, detail="No data available for the specified period")
        
        # Initialize charting system
        charting_system = AdvancedChartingSystem(data)
        
        # Create chart with indicators
        indicator_config = {
            'volume': 'volume' in (indicators or []),
            'macd': 'macd' in (indicators or []),
            'rsi': 'rsi' in (indicators or []),
            'bollinger': 'bollinger' in (indicators or []),
            'ichimoku': 'ichimoku' in (indicators or []),
            'fibonacci': 'fibonacci' in (indicators or [])
        }
        
        chart_data = charting_system.create_advanced_chart(indicator_config)
        return {
            "chart_data": chart_data,
            "patterns": charting_system.recognize_patterns()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 