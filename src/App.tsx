import React from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box, Container, Grid, TextField, Paper } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import StockChart from './components/StockChart';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

const queryClient = new QueryClient();

function App() {
  const [symbol, setSymbol] = React.useState('AAPL');

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Container maxWidth="lg">
          <Box sx={{ my: 4 }}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Paper sx={{ p: 2, mb: 2 }}>
                  <TextField
                    label="Stock Symbol"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                    sx={{ width: 200 }}
                  />
                </Paper>
              </Grid>
              <Grid item xs={12}>
                <StockChart symbol={symbol} />
              </Grid>
            </Grid>
          </Box>
        </Container>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App; 