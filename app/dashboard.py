"""
Streamlit Dashboard for Stock Market Prediction
Shows stock data, predictions, sentiment trends, and model performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Indian Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load processed features data"""
    try:
        df = pd.read_csv('data/processed/features.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("Data not found. Please run data collection and feature engineering first.")
        return None

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        with open('models/xgb_technical.pkl', 'rb') as f:
            model_tech = pickle.load(f)
        with open('models/xgb_hybrid.pkl', 'rb') as f:
            model_hybrid = pickle.load(f)
        return model_tech, model_hybrid
    except FileNotFoundError:
        st.error("Models not found. Please run model training first.")
        return None, None

@st.cache_data
def load_comparison():
    """Load model comparison data"""
    try:
        return pd.read_csv('models/model_comparison.csv')
    except FileNotFoundError:
        return None

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Indian Stock Market Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    model_tech, model_hybrid = load_models()
    
    # Sidebar - Stock selector
    st.sidebar.header("🎯 Stock Selection")
    
    available_symbols = sorted(df['Symbol'].unique())
    selected_symbol = st.sidebar.selectbox(
        "Select Stock Symbol",
        available_symbols,
        index=0
    )
    
    # Filter data for selected symbol
    symbol_df = df[df['Symbol'] == selected_symbol].copy()
    symbol_df = symbol_df.sort_values('Date')
    
    # Sidebar - Date range
    st.sidebar.header("📅 Date Range")
    min_date = symbol_df['Date'].min().date()
    max_date = symbol_df['Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(max_date - pd.Timedelta(days=90), max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        symbol_df = symbol_df[
            (symbol_df['Date'].dt.date >= start_date) & 
            (symbol_df['Date'].dt.date <= end_date)
        ]
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Price",
            f"₹{symbol_df['Close'].iloc[-1]:.2f}" if len(symbol_df) > 0 else "N/A",
            f"{symbol_df['Price_Change'].iloc[-1]*100:.2f}%" if len(symbol_df) > 0 else "N/A"
        )
    
    with col2:
        st.metric(
            "Total Records",
            len(symbol_df)
        )
    
    with col3:
        avg_sentiment = symbol_df['sentiment_score'].mean()
        sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
        st.metric(
            "Avg Sentiment",
            sentiment_label,
            f"{avg_sentiment:.3f}"
        )
    
    # Tab layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Stock Data", 
        "📈 Technical Indicators",
        "🤖 Predictions",
        "💭 Sentiment Analysis",
        "🎯 Model Performance"
    ])
    
    # Tab 1: Stock Data
    with tab1:
        st.subheader(f"Stock Price History - {selected_symbol}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(symbol_df['Date'], symbol_df['Close'], label='Close Price', linewidth=2)
        ax.fill_between(symbol_df['Date'], symbol_df['Low'], symbol_df['High'], alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (₹)')
        ax.set_title(f'{selected_symbol} Stock Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Recent Data")
        st.dataframe(
            symbol_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(10),
            use_container_width=True
        )
    
    # Tab 2: Technical Indicators
    with tab2:
        st.subheader("Technical Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(symbol_df['Date'], symbol_df['RSI'], label='RSI', color='purple')
            ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
            ax.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
            ax.set_xlabel('Date')
            ax.set_ylabel('RSI')
            ax.set_title('Relative Strength Index (RSI)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # MACD
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(symbol_df['Date'], symbol_df['MACD'], label='MACD', color='blue')
            ax.plot(symbol_df['Date'], symbol_df['MACD_signal'], label='Signal', color='red')
            ax.bar(symbol_df['Date'], symbol_df['MACD_diff'], label='Histogram', alpha=0.3)
            ax.set_xlabel('Date')
            ax.set_ylabel('MACD')
            ax.set_title('MACD Indicator')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Moving Averages
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(symbol_df['Date'], symbol_df['Close'], label='Close', linewidth=2)
            ax.plot(symbol_df['Date'], symbol_df['EMA_20'], label='EMA 20', linestyle='--')
            ax.plot(symbol_df['Date'], symbol_df['SMA_50'], label='SMA 50', linestyle='--')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (₹)')
            ax.set_title('Moving Averages')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Bollinger Bands
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(symbol_df['Date'], symbol_df['Close'], label='Close', linewidth=2)
            ax.plot(symbol_df['Date'], symbol_df['BB_high'], label='Upper Band', linestyle='--', color='red')
            ax.plot(symbol_df['Date'], symbol_df['BB_mid'], label='Middle Band', linestyle='--', color='gray')
            ax.plot(symbol_df['Date'], symbol_df['BB_low'], label='Lower Band', linestyle='--', color='green')
            ax.fill_between(symbol_df['Date'], symbol_df['BB_low'], symbol_df['BB_high'], alpha=0.1)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (₹)')
            ax.set_title('Bollinger Bands')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    
    # Tab 3: Predictions
    with tab3:
        st.subheader("Model Predictions")
        
        if model_tech is not None and model_hybrid is not None:
            # Prepare features
            technical_features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
                'EMA_20', 'SMA_50',
                'BB_high', 'BB_low', 'BB_mid',
                'Price_Change', 'Volume_Change'
            ]
            
            X_tech = symbol_df[technical_features].copy()
            X_hybrid = symbol_df[technical_features + ['sentiment_score']].copy()
            
            # Make predictions
            pred_tech = model_tech.predict(X_tech)
            pred_hybrid = model_hybrid.predict(X_hybrid)
            
            symbol_df['pred_technical'] = pred_tech
            symbol_df['pred_hybrid'] = pred_hybrid
            symbol_df['pred_tech_label'] = symbol_df['pred_technical'].map({1: '↑ Up', 0: '↓ Down'})
            symbol_df['pred_hybrid_label'] = symbol_df['pred_hybrid'].map({1: '↑ Up', 0: '↓ Down'})
            symbol_df['actual_label'] = symbol_df['target'].map({1: '↑ Up', 0: '↓ Down'})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Technical Model Predictions")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Plot actual vs predicted
                dates = symbol_df['Date'].values
                actual = symbol_df['target'].values
                predicted = symbol_df['pred_technical'].values
                
                ax.plot(dates, actual, 'o-', label='Actual', alpha=0.7, markersize=4)
                ax.plot(dates, predicted, 's-', label='Predicted', alpha=0.7, markersize=4)
                ax.set_xlabel('Date')
                ax.set_ylabel('Direction (0=Down, 1=Up)')
                ax.set_title('Technical Model: Predicted vs Actual')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Accuracy
                accuracy_tech = (symbol_df['pred_technical'] == symbol_df['target']).mean()
                st.metric("Accuracy", f"{accuracy_tech*100:.2f}%")
            
            with col2:
                st.markdown("### Hybrid Model Predictions")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Plot actual vs predicted
                predicted_hybrid = symbol_df['pred_hybrid'].values
                
                ax.plot(dates, actual, 'o-', label='Actual', alpha=0.7, markersize=4)
                ax.plot(dates, predicted_hybrid, 's-', label='Predicted', alpha=0.7, markersize=4)
                ax.set_xlabel('Date')
                ax.set_ylabel('Direction (0=Down, 1=Up)')
                ax.set_title('Hybrid Model: Predicted vs Actual')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Accuracy
                accuracy_hybrid = (symbol_df['pred_hybrid'] == symbol_df['target']).mean()
                st.metric("Accuracy", f"{accuracy_hybrid*100:.2f}%")
            
            st.subheader("Recent Predictions")
            st.dataframe(
                symbol_df[['Date', 'Close', 'actual_label', 'pred_tech_label', 'pred_hybrid_label']].tail(10),
                use_container_width=True
            )
        else:
            st.warning("Models not loaded. Please train models first.")
    
    # Tab 4: Sentiment Analysis
    with tab4:
        st.subheader("Sentiment Trend Analysis")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(symbol_df['Date'], symbol_df['sentiment_score'], linewidth=2, color='green')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(symbol_df['Date'], 0, symbol_df['sentiment_score'], 
                        where=(symbol_df['sentiment_score'] >= 0), alpha=0.3, color='green', label='Positive')
        ax.fill_between(symbol_df['Date'], 0, symbol_df['sentiment_score'], 
                        where=(symbol_df['sentiment_score'] < 0), alpha=0.3, color='red', label='Negative')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('News Sentiment Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            positive_days = (symbol_df['sentiment_score'] > 0.1).sum()
            st.metric("Positive Days", positive_days)
        
        with col2:
            neutral_days = ((symbol_df['sentiment_score'] >= -0.1) & (symbol_df['sentiment_score'] <= 0.1)).sum()
            st.metric("Neutral Days", neutral_days)
        
        with col3:
            negative_days = (symbol_df['sentiment_score'] < -0.1).sum()
            st.metric("Negative Days", negative_days)
    
    # Tab 5: Model Performance
    with tab5:
        st.subheader("Model Performance Comparison")
        
        comparison = load_comparison()
        if comparison is not None:
            st.dataframe(comparison, use_container_width=True)
            
            # Plot comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            x = np.arange(len(metrics))
            width = 0.35
            
            tech_values = comparison.iloc[0][metrics].values
            hybrid_values = comparison.iloc[1][metrics].values
            
            ax.bar(x - width/2, tech_values, width, label='Technical Only', color='skyblue')
            ax.bar(x + width/2, hybrid_values, width, label='Hybrid', color='lightgreen')
            
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show confusion matrices
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Technical Model")
                try:
                    img = plt.imread('models/confusion_matrix_technical.png')
                    st.image(img, use_container_width=True)
                except:
                    st.info("Confusion matrix image not found")
            
            with col2:
                st.markdown("#### Hybrid Model")
                try:
                    img = plt.imread('models/confusion_matrix_hybrid.png')
                    st.image(img, use_container_width=True)
                except:
                    st.info("Confusion matrix image not found")
        else:
            st.warning("Model comparison data not found. Please train models first.")

if __name__ == "__main__":
    main()
