"""
Data Collection Script for Indian Stock Market
Collects stock data from Yahoo Finance and news data from NewsAPI
"""

import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import os

# Configuration
NEWS_API_KEY = "135472927f3147439450e47924c2523b"

# Indian Stock Symbols (NSE)
SYMBOLS = [
    "RELIANCE.NS",  # Reliance Industries
    "TCS.NS",       # Tata Consultancy Services
    "INFY.NS",      # Infosys
    "^NSEI"         # Nifty 50 Index
]

def collect_stock_data():
    """Collect stock data from Yahoo Finance"""
    print("Collecting stock data from Yahoo Finance...")
    
    all_data = []
    
    for symbol in SYMBOLS:
        print(f"Fetching data for {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            # Get last 3 years of data
            df = ticker.history(period="3y", interval="1d")
            
            # Reset index to get date as a column
            df = df.reset_index()
            
            # Add symbol column
            df['Symbol'] = symbol
            
            # Select required columns
            df = df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            all_data.append(df)
            print(f"  → Collected {len(df)} records for {symbol}")
        except Exception as e:
            print(f"  ✗ Error fetching {symbol}: {e}")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date
    combined_df = combined_df.sort_values(['Symbol', 'Date'])
    
    # Save to CSV
    os.makedirs('data/raw', exist_ok=True)
    output_path = 'data/raw/technical_data.csv'
    combined_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Stock data saved to {output_path}")
    print(f"  Total records: {len(combined_df)}")
    print(f"  Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    
    return combined_df

def collect_news_data():
    """Collect news data from NewsAPI"""
    print("\nCollecting news data from NewsAPI...")
    
    # Initialize NewsAPI client
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    
    all_articles = []
    
    # Company keywords for Indian companies
    companies = {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services OR TCS",
        "INFY.NS": "Infosys",
        "^NSEI": "Nifty OR NSE OR Indian stock market"
    }
    
    # Get news from last 30 days (NewsAPI free tier limitation)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for symbol, query in companies.items():
        print(f"Fetching news for {symbol} ({query})...")
        
        try:
            # Search for news articles
            articles = newsapi.get_everything(
                q=query,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt',
                page_size=100
            )
            
            for article in articles.get('articles', []):
                all_articles.append({
                    'date': article['publishedAt'][:10],  # Get just the date
                    'symbol': symbol,
                    'headline': article['title'],
                    'description': article.get('description', ''),
                    'source': article['source']['name']
                })
            
            print(f"  → Collected {len(articles.get('articles', []))} articles")
        except Exception as e:
            print(f"  ✗ Error fetching news for {symbol}: {e}")
    
    # Create DataFrame
    news_df = pd.DataFrame(all_articles)
    
    if len(news_df) > 0:
        # Sort by date
        news_df = news_df.sort_values(['symbol', 'date'])
        
        # Save to CSV
        output_path = 'data/raw/news_data.csv'
        news_df.to_csv(output_path, index=False)
        
        print(f"\n✓ News data saved to {output_path}")
        print(f"  Total articles: {len(news_df)}")
        print(f"  Date range: {news_df['date'].min()} to {news_df['date'].max()}")
    else:
        print("\n⚠ Warning: No news articles collected")
        # Create empty file with headers
        news_df = pd.DataFrame(columns=['date', 'symbol', 'headline', 'description', 'source'])
        news_df.to_csv('data/raw/news_data.csv', index=False)
    
    return news_df

if __name__ == "__main__":
    print("=" * 60)
    print("Stock Market Data Collection - Indian Markets")
    print("=" * 60)
    
    # Collect stock data
    stock_df = collect_stock_data()
    
    # Collect news data
    news_df = collect_news_data()
    
    print("\n" + "=" * 60)
    print("Data collection completed!")
    print("=" * 60)
    print(f"\nStock data: {len(stock_df)} records")
    print(f"News data: {len(news_df)} articles")
    print("\nFiles created:")
    print("  - data/raw/technical_data.csv")
    print("  - data/raw/news_data.csv")
