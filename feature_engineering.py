"""
Feature Engineering Script
Computes technical indicators and sentiment analysis
"""

import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore')

def compute_technical_indicators(df):
    """Compute technical indicators for each symbol"""
    print("Computing technical indicators...")
    
    result_dfs = []
    
    for symbol in df['Symbol'].unique():
        print(f"Processing {symbol}...")
        symbol_df = df[df['Symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('Date')
        
        # RSI (14)
        rsi = RSIIndicator(close=symbol_df['Close'], window=14)
        symbol_df['RSI'] = rsi.rsi()
        
        # MACD (12, 26, 9)
        macd = MACD(close=symbol_df['Close'], window_slow=26, window_fast=12, window_sign=9)
        symbol_df['MACD'] = macd.macd()
        symbol_df['MACD_signal'] = macd.macd_signal()
        symbol_df['MACD_diff'] = macd.macd_diff()
        
        # EMA (20)
        ema = EMAIndicator(close=symbol_df['Close'], window=20)
        symbol_df['EMA_20'] = ema.ema_indicator()
        
        # SMA (50)
        sma = SMAIndicator(close=symbol_df['Close'], window=50)
        symbol_df['SMA_50'] = sma.sma_indicator()
        
        # Bollinger Bands
        bollinger = BollingerBands(close=symbol_df['Close'], window=20, window_dev=2)
        symbol_df['BB_high'] = bollinger.bollinger_hband()
        symbol_df['BB_low'] = bollinger.bollinger_lband()
        symbol_df['BB_mid'] = bollinger.bollinger_mavg()
        
        # Additional features
        symbol_df['Price_Change'] = symbol_df['Close'].pct_change()
        symbol_df['Volume_Change'] = symbol_df['Volume'].pct_change()
        
        result_dfs.append(symbol_df)
    
    combined_df = pd.concat(result_dfs, ignore_index=True)
    print(f"✓ Technical indicators computed for {len(combined_df)} records")
    
    return combined_df

def preprocess_and_get_sentiment():
    """Preprocess news data and compute sentiment scores"""
    print("\nProcessing news sentiment...")
    
    try:
        news_df = pd.read_csv('data/raw/news_data.csv')
        
        if len(news_df) == 0:
            print("⚠ No news data available, skipping sentiment analysis")
            return pd.DataFrame(columns=['date', 'symbol', 'sentiment_score'])
        
        print(f"Loaded {len(news_df)} news articles")
        
        # Import FinBERT
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        print("Loading FinBERT model...")
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        
        def get_sentiment_score(text):
            """Get sentiment score from FinBERT"""
            if pd.isna(text) or text == '':
                return 0.0
            
            try:
                # Tokenize
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Convert to sentiment score: positive=1, neutral=0, negative=-1
                # predictions[0]: [negative, neutral, positive]
                score = predictions[0][2].item() - predictions[0][0].item()
                return score
            except Exception as e:
                print(f"Error processing text: {e}")
                return 0.0
        
        # Combine headline and description
        news_df['text'] = news_df['headline'].fillna('') + ' ' + news_df['description'].fillna('')
        
        # Calculate sentiment for each article
        print("Calculating sentiment scores...")
        news_df['sentiment'] = news_df['text'].apply(lambda x: get_sentiment_score(x))
        
        # Aggregate daily mean sentiment per symbol
        sentiment_df = news_df.groupby(['date', 'symbol'])['sentiment'].mean().reset_index()
        sentiment_df.columns = ['date', 'symbol', 'sentiment_score']
        
        print(f"✓ Sentiment scores computed for {len(sentiment_df)} date-symbol pairs")
        
        return sentiment_df
        
    except FileNotFoundError:
        print("⚠ News data file not found, skipping sentiment analysis")
        return pd.DataFrame(columns=['date', 'symbol', 'sentiment_score'])
    except Exception as e:
        print(f"⚠ Error in sentiment analysis: {e}")
        return pd.DataFrame(columns=['date', 'symbol', 'sentiment_score'])

def create_target_variable(df):
    """Create target variable for prediction"""
    print("\nCreating target variable...")
    
    result_dfs = []
    
    for symbol in df['Symbol'].unique():
        symbol_df = df[df['Symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('Date')
        
        # Create target: 1 if next day close > today close, else 0
        symbol_df['next_close'] = symbol_df['Close'].shift(-1)
        symbol_df['target'] = (symbol_df['next_close'] > symbol_df['Close']).astype(int)
        
        result_dfs.append(symbol_df)
    
    combined_df = pd.concat(result_dfs, ignore_index=True)
    
    print(f"✓ Target variable created")
    print(f"  Class distribution: {combined_df['target'].value_counts().to_dict()}")
    
    return combined_df

def merge_features():
    """Main function to merge all features"""
    print("=" * 60)
    print("Feature Engineering - Indian Stock Market")
    print("=" * 60)
    
    # Load technical data
    print("\nLoading technical data...")
    tech_df = pd.read_csv('data/raw/technical_data.csv')
    tech_df['Date'] = pd.to_datetime(tech_df['Date'])
    print(f"Loaded {len(tech_df)} records")
    
    # Compute technical indicators
    tech_df = compute_technical_indicators(tech_df)
    
    # Get sentiment scores
    sentiment_df = preprocess_and_get_sentiment()
    
    # Merge technical and sentiment data
    if len(sentiment_df) > 0:
        print("\nMerging technical and sentiment data...")
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Rename columns for merge
        sentiment_df['Date'] = sentiment_df['date']
        sentiment_df['Symbol'] = sentiment_df['symbol']
        
        merged_df = tech_df.merge(
            sentiment_df[['Date', 'Symbol', 'sentiment_score']], 
            on=['Date', 'Symbol'], 
            how='left'
        )
        
        # Fill missing sentiment with 0 (neutral)
        merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(0)
        
        print(f"✓ Merged data: {len(merged_df)} records")
    else:
        print("\n⚠ No sentiment data available, using technical indicators only")
        merged_df = tech_df.copy()
        merged_df['sentiment_score'] = 0
    
    # Create target variable
    merged_df = create_target_variable(merged_df)
    
    # Drop rows with missing values
    print("\nCleaning data...")
    print(f"  Before cleaning: {len(merged_df)} records")
    merged_df = merged_df.dropna()
    print(f"  After cleaning: {len(merged_df)} records")
    
    # Save processed data
    output_path = 'data/processed/features.csv'
    merged_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Features saved to {output_path}")
    print("\nFeature columns:")
    for col in merged_df.columns:
        print(f"  - {col}")
    
    return merged_df

if __name__ == "__main__":
    df = merge_features()
    print("\n" + "=" * 60)
    print("Feature engineering completed!")
    print("=" * 60)
