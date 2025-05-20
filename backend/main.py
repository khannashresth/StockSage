from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

app = FastAPI(
    title="Stock Market Analysis API",
    description="Backend API for Stock Market Analysis Platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routes
from routes import stocks, portfolio, analysis, backtesting, sentiment

# Include routers
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(backtesting.router, prefix="/api/backtesting", tags=["backtesting"])
app.include_router(sentiment.router, prefix="/api/sentiment", tags=["sentiment"])

@app.get("/")
async def root():
    return {
        "message": "Stock Market Analysis API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 