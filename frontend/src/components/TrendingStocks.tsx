import { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  CircularProgress,
  Chip
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import axios from 'axios';
import { TrendingStock } from '../types';

const TrendingStocks = () => {
  const [trending, setTrending] = useState<TrendingStock[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTrending = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await axios.get('http://localhost:5000/api/market/trending');
        if (response.data.success) {
          setTrending(response.data.trending);
        } else {
          setError('Failed to fetch trending stocks');
        }
      } catch (err) {
        setError('Error connecting to server');
      } finally {
        setLoading(false);
      }
    };

    fetchTrending();
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Typography color="error" align="center">
        {error}
      </Typography>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Trending Stocks
      </Typography>
      <Grid container spacing={2}>
        {trending.map((stock) => (
          <Grid item xs={12} sm={6} md={4} key={stock.symbol}>
            <Paper
              sx={{
                p: 2,
                display: 'flex',
                flexDirection: 'column',
                height: '100%',
              }}
            >
              <Typography variant="subtitle1" component="div">
                {stock.name}
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {stock.symbol}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                <Chip
                  icon={stock.change >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
                  label={`${stock.change >= 0 ? '+' : ''}${stock.change.toFixed(2)}%`}
                  color={stock.change >= 0 ? 'success' : 'error'}
                  size="small"
                />
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default TrendingStocks; 