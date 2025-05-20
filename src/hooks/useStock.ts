import { useQuery } from '@tanstack/react-query';
import { api, StockData, MarketDepth } from '../services/api';

export const useStockData = (symbol: string, timeframe: string = '1d') => {
  return useQuery<StockData[]>({
    queryKey: ['stock', symbol, timeframe],
    queryFn: () => api.getStockData(symbol, timeframe),
  });
};

export const useMarketDepth = (symbol: string) => {
  return useQuery<MarketDepth>({
    queryKey: ['marketDepth', symbol],
    queryFn: () => api.getMarketDepth(symbol),
    refetchInterval: 5000, // Refresh every 5 seconds
  });
};

export const useStockChart = (
  symbol: string,
  startDate: Date,
  endDate?: Date,
  indicators?: string[]
) => {
  return useQuery({
    queryKey: ['stockChart', symbol, startDate, endDate, indicators],
    queryFn: () => api.getStockChart(symbol, startDate, endDate, indicators),
  });
};

export const useStockSearch = (query: string) => {
  return useQuery({
    queryKey: ['stockSearch', query],
    queryFn: () => api.searchStocks(query),
    enabled: query.length > 0,
  });
}; 