import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
from textblob import TextBlob
import re
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import os
from dotenv import load_dotenv
import base64
import yfinance as yf

# Load environment variables
load_dotenv()

class NewsAnalyzer:
    def __init__(self):
        self.news_sources = {
            'Reuters': 'https://www.reuters.com/markets/companies',
            'Bloomberg': 'https://www.bloomberg.com/markets',
            'Financial Times': 'https://www.ft.com/markets',
            'Economic Times': 'https://economictimes.indiatimes.com/markets'
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.logger = logging.getLogger('NewsAnalyzer')
    
    def get_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Get news articles for a symbol from multiple sources."""
        news_articles = []
        
        # Get news from various sources
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for source, url in self.news_sources.items():
                futures.append(
                    executor.submit(self._fetch_news, source, url, symbol, days)
                )
            
            for future in futures:
                try:
                    articles = future.result()
                    news_articles.extend(articles)
                except Exception as e:
                    self.logger.error(f"Error fetching news: {str(e)}")
        
        # Sort by date
        news_articles.sort(key=lambda x: x['date'], reverse=True)
        return news_articles
    
    def _fetch_news(self, source: str, url: str, symbol: str, days: int) -> List[Dict]:
        """Fetch news from a specific source."""
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = []
                
                # Extract articles based on source-specific selectors
                if source == 'Reuters':
                    articles = self._parse_reuters(soup, symbol, days)
                elif source == 'Bloomberg':
                    articles = self._parse_bloomberg(soup, symbol, days)
                elif source == 'Financial Times':
                    articles = self._parse_ft(soup, symbol, days)
                elif source == 'Economic Times':
                    articles = self._parse_et(soup, symbol, days)
                
                return articles
            return []
        except Exception as e:
            self.logger.error(f"Error fetching from {source}: {str(e)}")
            return []
    
    def _parse_reuters(self, soup: BeautifulSoup, symbol: str, days: int) -> List[Dict]:
        """Parse Reuters articles."""
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for article in soup.find_all('article'):
            try:
                title = article.find('h3').text.strip()
                if symbol.lower() in title.lower():
                    date_str = article.find('time')['datetime']
                    date = datetime.fromisoformat(date_str)
                    
                    if date >= cutoff_date:
                        articles.append({
                            'source': 'Reuters',
                            'title': title,
                            'url': 'https://www.reuters.com' + article.find('a')['href'],
                            'date': date,
                            'sentiment': self._analyze_sentiment(title)
                        })
            except:
                continue
        
        return articles
    
    def _parse_bloomberg(self, soup: BeautifulSoup, symbol: str, days: int) -> List[Dict]:
        """Parse Bloomberg articles."""
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for article in soup.find_all('article'):
            try:
                title = article.find('h3').text.strip()
                if symbol.lower() in title.lower():
                    date_str = article.find('time')['datetime']
                    date = datetime.fromisoformat(date_str)
                    
                    if date >= cutoff_date:
                        articles.append({
                            'source': 'Bloomberg',
                            'title': title,
                            'url': article.find('a')['href'],
                            'date': date,
                            'sentiment': self._analyze_sentiment(title)
                        })
            except:
                continue
        
        return articles
    
    def _parse_ft(self, soup: BeautifulSoup, symbol: str, days: int) -> List[Dict]:
        """Parse Financial Times articles."""
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for article in soup.find_all('div', class_='o-teaser'):
            try:
                title = article.find('a').text.strip()
                if symbol.lower() in title.lower():
                    date_str = article.find('time')['datetime']
                    date = datetime.fromisoformat(date_str)
                    
                    if date >= cutoff_date:
                        articles.append({
                            'source': 'Financial Times',
                            'title': title,
                            'url': 'https://www.ft.com' + article.find('a')['href'],
                            'date': date,
                            'sentiment': self._analyze_sentiment(title)
                        })
            except:
                continue
        
        return articles
    
    def _parse_et(self, soup: BeautifulSoup, symbol: str, days: int) -> List[Dict]:
        """Parse Economic Times articles."""
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for article in soup.find_all('div', class_='eachStory'):
            try:
                title = article.find('h3').text.strip()
                if symbol.lower() in title.lower():
                    date_str = article.find('time')['datetime']
                    date = datetime.fromisoformat(date_str)
                    
                    if date >= cutoff_date:
                        articles.append({
                            'source': 'Economic Times',
                            'title': title,
                            'url': article.find('a')['href'],
                            'date': date,
                            'sentiment': self._analyze_sentiment(title)
                        })
            except:
                continue
        
        return articles
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using TextBlob."""
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
    
    def plot_sentiment_trends(self, articles: List[Dict]) -> go.Figure:
        """Plot sentiment trends over time."""
        df = pd.DataFrame(articles)
        df['date'] = pd.to_datetime(df['date'])
        df['polarity'] = df['sentiment'].apply(lambda x: x['polarity'])
        df['subjectivity'] = df['sentiment'].apply(lambda x: x['subjectivity'])
        
        # Calculate daily average sentiment
        daily_sentiment = df.groupby('date')[['polarity', 'subjectivity']].mean()
        
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=('Sentiment Polarity', 'Sentiment Subjectivity'),
            shared_xaxes=True
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_sentiment.index,
                y=daily_sentiment['polarity'],
                mode='lines+markers',
                name='Polarity',
                line=dict(color='blue')
            ),
            row=1,
            col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_sentiment.index,
                y=daily_sentiment['subjectivity'],
                mode='lines+markers',
                name='Subjectivity',
                line=dict(color='red')
            ),
            row=2,
            col=1
        )
        
        fig.update_layout(
            title='News Sentiment Trends',
            height=800,
            showlegend=True
        )
        
        return fig

class SocialMediaAnalyzer:
    def __init__(self):
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.logger = logging.getLogger('SocialMediaAnalyzer')
    
    def get_social_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """Get sentiment from social media platforms."""
        sentiment = {
            'twitter': self._get_twitter_sentiment(symbol, days),
            'reddit': self._get_reddit_sentiment(symbol, days)
        }
        return sentiment
    
    def _get_twitter_sentiment(self, symbol: str, days: int) -> List[Dict]:
        """Get sentiment from Twitter."""
        if not self.twitter_api_key or not self.twitter_api_secret:
            return []
        
        try:
            # Twitter API v2 endpoint
            url = "https://api.twitter.com/2/tweets/search/recent"
            
            # Get bearer token
            auth_url = "https://api.twitter.com/oauth2/token"
            auth_headers = {
                'Authorization': f"Basic {base64.b64encode(f'{self.twitter_api_key}:{self.twitter_api_secret}'.encode()).decode()}"
            }
            auth_data = {
                'grant_type': 'client_credentials'
            }
            
            auth_response = requests.post(auth_url, headers=auth_headers, data=auth_data)
            bearer_token = auth_response.json()['access_token']
            
            # Search tweets
            headers = {
                'Authorization': f"Bearer {bearer_token}"
            }
            
            params = {
                'query': f"${symbol} lang:en -is:retweet",
                'max_results': 100,
                'tweet.fields': 'created_at,public_metrics'
            }
            
            response = requests.get(url, headers=headers, params=params)
            tweets = response.json()['data']
            
            # Analyze sentiment
            results = []
            for tweet in tweets:
                sentiment = self._analyze_sentiment(tweet['text'])
                results.append({
                    'text': tweet['text'],
                    'date': datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                    'metrics': tweet['public_metrics'],
                    'sentiment': sentiment
                })
            
            return results
        except Exception as e:
            self.logger.error(f"Error fetching Twitter data: {str(e)}")
            return []
    
    def _get_reddit_sentiment(self, symbol: str, days: int) -> List[Dict]:
        """Get sentiment from Reddit."""
        if not self.reddit_client_id or not self.reddit_client_secret:
            return []
        
        try:
            # Reddit API authentication
            auth = requests.auth.HTTPBasicAuth(self.reddit_client_id, self.reddit_client_secret)
            
            # Get access token
            data = {
                'grant_type': 'client_credentials',
                'username': os.getenv('REDDIT_USERNAME'),
                'password': os.getenv('REDDIT_PASSWORD')
            }
            
            headers = {'User-Agent': 'MyAPI/0.0.1'}
            
            response = requests.post(
                'https://www.reddit.com/api/v1/access_token',
                auth=auth,
                data=data,
                headers=headers
            )
            
            token = response.json()['access_token']
            headers['Authorization'] = f"bearer {token}"
            
            # Search posts
            subreddits = ['wallstreetbets', 'stocks', 'investing']
            results = []
            
            for subreddit in subreddits:
                response = requests.get(
                    f"https://oauth.reddit.com/r/{subreddit}/search",
                    headers=headers,
                    params={'q': symbol, 'limit': 100, 't': 'week'}
                )
                
                posts = response.json()['data']['children']
                
                for post in posts:
                    data = post['data']
                    sentiment = self._analyze_sentiment(data['title'] + ' ' + data['selftext'])
                    results.append({
                        'title': data['title'],
                        'text': data['selftext'],
                        'date': datetime.fromtimestamp(data['created_utc']),
                        'score': data['score'],
                        'sentiment': sentiment
                    })
            
            return results
        except Exception as e:
            self.logger.error(f"Error fetching Reddit data: {str(e)}")
            return []
    
    def plot_social_sentiment(self, sentiment_data: Dict) -> go.Figure:
        """Plot social media sentiment analysis."""
        # Process Twitter data
        twitter_df = pd.DataFrame(sentiment_data['twitter'])
        if not twitter_df.empty:
            twitter_df['date'] = pd.to_datetime(twitter_df['date'])
            twitter_df['polarity'] = twitter_df['sentiment'].apply(lambda x: x['polarity'])
        
        # Process Reddit data
        reddit_df = pd.DataFrame(sentiment_data['reddit'])
        if not reddit_df.empty:
            reddit_df['date'] = pd.to_datetime(reddit_df['date'])
            reddit_df['polarity'] = reddit_df['sentiment'].apply(lambda x: x['polarity'])
        
        # Create figure
        fig = go.Figure()

        # TradingView-style layout
        layout = dict(
            plot_bgcolor='rgb(19,23,34)',
            paper_bgcolor='rgb(19,23,34)',
            font=dict(color='rgb(255,255,255)', size=12),
            title=dict(
                text='Social Media Sentiment Analysis',
                font=dict(size=20, color='rgb(255,255,255)'),
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                tickfont=dict(size=12),
                linecolor='rgba(128,128,128,0.2)',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                tickfont=dict(size=12),
                linecolor='rgba(128,128,128,0.2)',
                zeroline=False,
                title=dict(text='Sentiment Score', font=dict(size=14))
            ),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgb(255,255,255)'),
                bordercolor='rgba(128,128,128,0.2)'
            ),
            hovermode='x unified',
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Add Twitter sentiment if data exists
        if not twitter_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=twitter_df['date'],
                    y=twitter_df['polarity'].rolling(window=10).mean(),
                    name='Twitter Sentiment',
                    line=dict(color='rgb(41,98,255)', width=2),
                    mode='lines',
                    hovertemplate='%{y:.2f}<extra>Twitter</extra>'
                )
            )

        # Add Reddit sentiment if data exists
        if not reddit_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=reddit_df['date'],
                    y=reddit_df['polarity'].rolling(window=10).mean(),
                    name='Reddit Sentiment',
                    line=dict(color='rgb(255,82,82)', width=2),
                    mode='lines',
                    hovertemplate='%{y:.2f}<extra>Reddit</extra>'
                )
            )

        # Update layout
        fig.update_layout(layout)
        
        return fig

class MarketSentimentAnalyzer:
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'V88RPGMHJIFNFSCK')
        self.news_analyzer = NewsAnalyzer()
        self.social_analyzer = SocialMediaAnalyzer()
        self.logger = logging.getLogger(__name__)

    def analyze_market_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """Analyze market sentiment using news from Yahoo Finance and Alpha Vantage."""
        try:
            # Get stock data from Yahoo Finance
            stock = yf.Ticker(symbol)
            yf_news = stock.news
            
            # Get news from Alpha Vantage
            av_news = self._get_alpha_vantage_news(symbol)
            
            # Combine news from both sources
            all_news = []
            
            # Process Yahoo Finance news
            for article in yf_news:
                try:
                    all_news.append({
                        'source': 'Yahoo Finance',
                        'title': article.get('title', ''),
                        'url': article.get('link', ''),
                        'date': datetime.fromtimestamp(article.get('providerPublishTime', 0)),
                        'sentiment': self._analyze_sentiment(article.get('title', '') + ' ' + article.get('summary', ''))
                    })
                except Exception as e:
                    self.logger.warning(f"Error processing Yahoo Finance news: {str(e)}")
            
            # Add Alpha Vantage news
            all_news.extend(av_news)
            
            # Get additional news from traditional sources
            news_articles = self.news_analyzer.get_news(symbol, days)
            all_news.extend(news_articles)
            
            # Get social media sentiment
            social_sentiment = self.social_analyzer.get_social_sentiment(symbol, days)
            
            # Calculate overall sentiment
            sentiments = [article['sentiment']['polarity'] for article in all_news]
            overall_sentiment = np.mean(sentiments) if sentiments else 0
            
            # Prepare response
            sentiment_data = {
                'overall_sentiment': overall_sentiment,
                'news_articles': sorted(all_news, key=lambda x: x['date'], reverse=True),
                'social_sentiment': social_sentiment,
                'analysis_date': datetime.now(),
                'symbol': symbol
            }
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                'overall_sentiment': 0,
                'news_articles': [],
                'social_sentiment': {'twitter': [], 'reddit': []},
                'analysis_date': datetime.now(),
                'symbol': symbol
            }

    def _get_alpha_vantage_news(self, symbol: str) -> List[Dict]:
        """Get news from Alpha Vantage API."""
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.alpha_vantage_key}"
            response = requests.get(url)
            data = response.json()
            
            news_articles = []
            for item in data.get('feed', []):
                try:
                    news_articles.append({
                        'source': 'Alpha Vantage',
                        'title': item.get('title', ''),
                        'url': item.get('url', ''),
                        'date': datetime.strptime(item.get('time_published', ''), '%Y%m%dT%H%M%S'),
                        'sentiment': {
                            'polarity': float(item.get('overall_sentiment_score', 0)),
                            'subjectivity': 0.5  # Alpha Vantage doesn't provide subjectivity
                        }
                    })
                except Exception as e:
                    self.logger.warning(f"Error processing Alpha Vantage news item: {str(e)}")
            
            return news_articles
            
        except Exception as e:
            self.logger.error(f"Error fetching Alpha Vantage news: {str(e)}")
            return []

    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using TextBlob."""
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }

    def plot_combined_sentiment(self, sentiment_data: Dict) -> go.Figure:
        """Create an advanced sentiment visualization."""
        try:
            # Convert news articles to DataFrame
            news_df = pd.DataFrame(sentiment_data['news_articles'])
            news_df['date'] = pd.to_datetime(news_df['date'])
            news_df['polarity'] = news_df['sentiment'].apply(lambda x: x['polarity'])
            
            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('News Sentiment Over Time', 'Sentiment Distribution'),
                vertical_spacing=0.2
            )
            
            # Add sentiment trend
            daily_sentiment = news_df.groupby(news_df['date'].dt.date)['polarity'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_sentiment.index,
                    y=daily_sentiment.values,
                    mode='lines+markers',
                    name='Daily Sentiment',
                    line=dict(color='#00d09c', width=2),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            # Add sentiment distribution
            fig.add_trace(
                go.Histogram(
                    x=news_df['polarity'],
                    name='Sentiment Distribution',
                    marker_color='#00d09c',
                    opacity=0.7,
                    nbinsx=20
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            # Update axes
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=2, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=2, col=1)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating sentiment plot: {str(e)}")
            return go.Figure() 