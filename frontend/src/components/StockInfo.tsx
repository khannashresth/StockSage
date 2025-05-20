import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  CircularProgress,
  Alert,
} from '@mui/material';
import axios from 'axios';
import type { StockResponse } from '../types';

interface StockInfoProps {
  symbol: string;
}

const StockInfo: React.FC<StockInfoProps> = ({ symbol }) => {
  const [data, setData] = useState<StockResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`http://localhost:5000/api/stock/${symbol}`);
        setData(response.data);
        setError(null);
      } catch (err) {
        setError('Failed to fetch stock information');
        console.error('Error fetching stock data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [symbol]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" p={3}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  if (!data) {
    return <Alert severity="info">No data available</Alert>;
  }

  const { info, stats } = data;

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Company Information
      </Typography>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            {info.name} ({symbol})
          </Typography>
          <Typography variant="body1" gutterBottom>
            Sector: {info.sector}
          </Typography>
          <Typography variant="body1" gutterBottom>
            Industry: {info.industry}
          </Typography>
          <Typography variant="body1" gutterBottom>
            Market Cap: ${(info.marketCap / 1e9).toFixed(2)}B
          </Typography>
          <Typography variant="body1" gutterBottom>
            P/E Ratio: {info.peRatio?.toFixed(2) || 'N/A'}
          </Typography>
          <Typography variant="body1" gutterBottom>
            EPS: ${info.eps?.toFixed(2) || 'N/A'}
          </Typography>
        </Paper>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Statistics
          </Typography>
          <Typography variant="body1" gutterBottom>
            Volatility: {stats.volatility.toFixed(2)}%
          </Typography>
          <Typography variant="body1" gutterBottom>
            Average Volume: {stats.avgVolume.toLocaleString()}
          </Typography>
          <Typography variant="body1" gutterBottom>
            Price Change: {stats.priceChange.toFixed(2)}%
          </Typography>
          <Typography variant="body1" gutterBottom>
            Current RSI: {stats.currentRSI.toFixed(2)}
          </Typography>
          <Typography variant="body1" gutterBottom>
            Current MACD: {stats.currentMACD.toFixed(2)}
          </Typography>
        </Paper>
      </Box>
    </Box>
  );
};

export default StockInfo; 