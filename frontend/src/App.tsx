import { useState } from 'react';
import {
  Container,
  TextField,
  Box,
  Typography,
  Autocomplete,
  Paper,
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  useMediaQuery
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import InfoIcon from '@mui/icons-material/Info';
import StockChart from './components/StockChart';
import StockInfo from './components/StockInfo';
import TrendingStocks from './components/TrendingStocks';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
  },
});

function App() {
  const [selectedStock, setSelectedStock] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const isMobile = useMediaQuery('(max-width:600px)');

  const [stockOptions] = useState([
    { label: 'Apple Inc.', symbol: 'AAPL' },
    { label: 'Microsoft Corporation', symbol: 'MSFT' },
    { label: 'Alphabet Inc.', symbol: 'GOOGL' },
    { label: 'Amazon.com Inc.', symbol: 'AMZN' },
    { label: 'Meta Platforms Inc.', symbol: 'META' },
    { label: 'Tesla Inc.', symbol: 'TSLA' },
    { label: 'NVIDIA Corporation', symbol: 'NVDA' },
    { label: 'JPMorgan Chase & Co.', symbol: 'JPM' },
    { label: 'Visa Inc.', symbol: 'V' },
    { label: 'Walmart Inc.', symbol: 'WMT' },
  ]);

  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <AppBar position="static">
          <Toolbar>
            <IconButton
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={toggleDrawer}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              StockSage
            </Typography>
          </Toolbar>
        </AppBar>

        <Drawer anchor="left" open={drawerOpen} onClose={toggleDrawer}>
          <Box sx={{ width: 250 }}>
            <List>
              <ListItem>
                <ListItemIcon>
                  <ShowChartIcon />
                </ListItemIcon>
                <ListItemText primary="Market Overview" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <TrendingUpIcon />
                </ListItemIcon>
                <ListItemText primary="Trending Stocks" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <InfoIcon />
                </ListItemIcon>
                <ListItemText primary="About" />
              </ListItem>
            </List>
          </Box>
        </Drawer>

        <Container maxWidth="lg" sx={{ mt: 4, mb: 4, flex: 1 }}>
          <Box sx={{ mb: 4 }}>
            <Typography variant="h4" component="h1" gutterBottom>
              Stock Market Dashboard
            </Typography>

            <Autocomplete
              options={stockOptions}
              getOptionLabel={(option) => `${option.label} (${option.symbol})`}
              onChange={(_, newValue) => setSelectedStock(newValue?.symbol || null)}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Search for a stock"
                  variant="outlined"
                  fullWidth
                  sx={{ mb: 4 }}
                />
              )}
            />

            {!selectedStock && (
              <Paper sx={{ p: 3, mb: 4 }}>
                <TrendingStocks />
              </Paper>
            )}

            {selectedStock && (
              <Box sx={{ display: 'grid', gap: 3 }}>
                <Paper sx={{ p: 3 }}>
                  <StockChart symbol={selectedStock} />
                </Paper>

                <Paper sx={{ p: 3 }}>
                  <StockInfo symbol={selectedStock} />
                </Paper>
              </Box>
            )}
          </Box>
        </Container>

        <Box
          component="footer"
          sx={{
            py: 3,
            px: 2,
            mt: 'auto',
            backgroundColor: (theme) =>
              theme.palette.mode === 'light'
                ? theme.palette.grey[200]
                : theme.palette.grey[800],
          }}
        >
          <Container maxWidth="sm">
            <Typography variant="body2" color="text.secondary" align="center">
              © {new Date().getFullYear()} StockSage. All rights reserved.
            </Typography>
          </Container>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
