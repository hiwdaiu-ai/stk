# Hybrid Stock Market Prediction using Technical and Sentiment Analysis

## Indian Markets Focus

This project implements a comprehensive stock market prediction system specifically targeting Indian markets (NSE). It combines technical indicators with sentiment analysis from news to predict stock price movements.

## 🎯 Project Overview

The system uses:
- **Technical Analysis**: RSI, MACD, EMA, SMA, Bollinger Bands
- **Sentiment Analysis**: FinBERT model for news sentiment scoring
- **Machine Learning**: XGBoost classifier with two approaches:
  - Model A: Technical indicators only
  - Model B: Hybrid (Technical + Sentiment)

## 📊 Indian Stocks Analyzed

- **RELIANCE.NS** - Reliance Industries Limited
- **TCS.NS** - Tata Consultancy Services
- **INFY.NS** - Infosys Limited
- **^NSEI** - Nifty 50 Index

## 🏗️ Project Structure

```
stock-prediction/
│
├── data/
│   ├── raw/                    # Raw data from APIs
│   │   ├── technical_data.csv  # Stock OHLCV data
│   │   └── news_data.csv       # News headlines and sentiment
│   └── processed/              # Processed features
│       └── features.csv        # Combined technical + sentiment features
│
├── models/                     # Trained models and results
│   ├── xgb_technical.pkl      # Technical-only model
│   ├── xgb_hybrid.pkl         # Hybrid model
│   ├── confusion_matrix_*.png
│   ├── feature_importance_*.png
│   └── model_comparison.csv
│
├── app/
│   └── dashboard.py           # Streamlit dashboard
│
├── collect_data.py            # Data collection script
├── feature_engineering.py     # Feature engineering script
├── train_models.py           # Model training script
├── run_pipeline.py           # Complete pipeline runner
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python run_pipeline.py
```

This will:
1. ✅ Collect stock data from Yahoo Finance (last 3 years)
2. ✅ Collect news data from NewsAPI (last 30 days)
3. ✅ Compute technical indicators
4. ✅ Perform sentiment analysis using FinBERT
5. ✅ Train both XGBoost models
6. ✅ Generate performance metrics and visualizations

### 3. View Dashboard

```bash
streamlit run app/dashboard.py
```

Then open http://localhost:8501 in your browser.

## 📋 Individual Scripts

You can also run each step individually:

### Data Collection
```bash
python collect_data.py
```
Collects:
- Stock OHLCV data for Indian stocks (3 years)
- News headlines from NewsAPI (30 days, API limitation)

### Feature Engineering
```bash
python feature_engineering.py
```
Computes:
- Technical indicators (RSI, MACD, EMA, SMA, Bollinger Bands)
- Sentiment scores using FinBERT
- Creates target variable (next day price direction)

### Model Training
```bash
python train_models.py
```
Trains:
- XGBoost with technical indicators only
- XGBoost with technical + sentiment features
- Generates performance comparisons and visualizations

## 📊 Dashboard Features

The Streamlit dashboard provides:

1. **Stock Selection**: Choose from available Indian stocks
2. **Stock Data Visualization**: Price history with volume
3. **Technical Indicators**: RSI, MACD, Moving Averages, Bollinger Bands
4. **Predictions**: Compare technical vs hybrid model predictions
5. **Sentiment Analysis**: News sentiment trends over time
6. **Model Performance**: Accuracy, precision, recall, F1-score comparisons

## 🔑 API Keys

The project uses:
- **Yahoo Finance**: No API key required (via yfinance library)
- **NewsAPI**: API key included in code (135472927f3147439450e47924c2523b)

## 📈 Expected Results

- Trained models with accuracy metrics
- Comparison showing sentiment data impact
- Interactive dashboard for analysis
- Visualizations of predictions and performance

## 🛠️ Technologies Used

- **Data Collection**: yfinance, newsapi-python
- **Data Processing**: pandas, numpy
- **Technical Indicators**: ta library
- **Sentiment Analysis**: transformers (FinBERT), torch
- **Machine Learning**: scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn, streamlit

## 📝 Notes

- NewsAPI free tier limits data to last 30 days
- Stock data covers last 3 years with daily intervals
- Models are trained on 70/15/15 train/validation/test split
- All prices are in Indian Rupees (₹)

## 🎓 Project Components

1. **Data Collection**: Real-time data from Yahoo Finance and NewsAPI
2. **Feature Engineering**: 16+ technical features + sentiment scores
3. **Model Training**: XGBoost with hyperparameter tuning
4. **Evaluation**: Comprehensive metrics and confusion matrices
5. **Visualization**: Interactive Streamlit dashboard
6. **Documentation**: Complete project report and code documentation

## 📄 License

This project is for educational and research purposes.

## 🔗 References

- Yahoo Finance API via yfinance
- NewsAPI for news data
- FinBERT model: yiyanghkust/finbert-tone (Hugging Face)
- XGBoost Documentation
- Technical Analysis Library (ta)
