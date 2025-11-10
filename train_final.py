"""
Final Improved Model Training with Ensemble and Better Strategy
Uses GridSearch CV for hyperparameter tuning and ensemble methods
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_engineer_features():
    """Load data and create all features"""
    print("Loading and engineering features...")
    
    df = pd.read_csv('data/processed/features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    result_dfs = []
    
    for symbol in df['Symbol'].unique():
        symbol_df = df[df['Symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('Date')
        
        # Momentum indicators
        for period in [3, 5, 10, 20]:
            symbol_df[f'returns_{period}d'] = symbol_df['Close'].pct_change(period)
            symbol_df[f'volume_change_{period}d'] = symbol_df['Volume'].pct_change(period)
        
        # Volatility
        symbol_df['volatility_10d'] = symbol_df['Close'].pct_change().rolling(10).std()
        symbol_df['volatility_20d'] = symbol_df['Close'].pct_change().rolling(20).std()
        
        # Price position relative to MAs
        symbol_df['price_vs_ema20'] = (symbol_df['Close'] - symbol_df['EMA_20']) / symbol_df['EMA_20']
        symbol_df['price_vs_sma50'] = (symbol_df['Close'] - symbol_df['SMA_50']) / symbol_df['SMA_50']
        
        # RSI features
        symbol_df['rsi_normalized'] = symbol_df['RSI'] / 100
        symbol_df['rsi_overbought'] = (symbol_df['RSI'] > 70).astype(int)
        symbol_df['rsi_oversold'] = (symbol_df['RSI'] < 30).astype(int)
        
        # MACD
        symbol_df['macd_signal_cross'] = ((symbol_df['MACD'] > symbol_df['MACD_signal']) & 
                                          (symbol_df['MACD'].shift(1) <= symbol_df['MACD_signal'].shift(1))).astype(int)
        
        # Bollinger Bands
        bb_width = symbol_df['BB_high'] - symbol_df['BB_low']
        symbol_df['bb_width'] = bb_width / symbol_df['Close']
        symbol_df['bb_position'] = (symbol_df['Close'] - symbol_df['BB_low']) / (bb_width + 1e-10)
        
        # Volume
        symbol_df['volume_ma20'] = symbol_df['Volume'].rolling(20).mean()
        symbol_df['volume_ratio'] = symbol_df['Volume'] / (symbol_df['volume_ma20'] + 1)
        
        # Trend
        symbol_df['close_ma5'] = symbol_df['Close'].rolling(5).mean()
        symbol_df['close_ma10'] = symbol_df['Close'].rolling(10).mean()
        symbol_df['trend_5_10'] = (symbol_df['close_ma5'] > symbol_df['close_ma10']).astype(int)
        
        # Price range
        symbol_df['hl_ratio'] = (symbol_df['High'] - symbol_df['Low']) / (symbol_df['Close'] + 1e-10)
        
        # Sentiment features
        symbol_df['sentiment_ma3'] = symbol_df['sentiment_score'].rolling(3).mean()
        symbol_df['sentiment_ma7'] = symbol_df['sentiment_score'].rolling(7).mean()
        symbol_df['sentiment_positive'] = (symbol_df['sentiment_score'] > 0).astype(int)
        
        result_dfs.append(symbol_df)
    
    combined_df = pd.concat(result_dfs, ignore_index=True)
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
    combined_df = combined_df.dropna()
    
    print(f"✓ Features engineered: {len(combined_df)} records")
    
    return combined_df

def prepare_datasets(df, include_sentiment=False):
    """Prepare train/val/test sets"""
    
    # Select features
    base_features = [
        'RSI', 'MACD', 'MACD_diff',
        'Price_Change', 'Volume_Change',
        'returns_3d', 'returns_5d', 'returns_10d', 'returns_20d',
        'volume_change_3d', 'volume_change_10d',
        'volatility_10d', 'volatility_20d',
        'price_vs_ema20', 'price_vs_sma50',
        'rsi_normalized', 'rsi_overbought', 'rsi_oversold',
        'macd_signal_cross',
        'bb_width', 'bb_position',
        'volume_ratio',
        'trend_5_10',
        'hl_ratio'
    ]
    
    if include_sentiment:
        features = base_features + ['sentiment_score', 'sentiment_ma3', 'sentiment_ma7', 'sentiment_positive']
    else:
        features = base_features
    
    X = df[features].values
    y = df['target'].values
    
    # Split: 70-15-15
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, shuffle=False)  # 0.176 * 0.85 ≈ 0.15
    
    # Scale features
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, features

def train_ensemble_model(X_train, y_train, X_val, y_val, model_name):
    """Train ensemble model"""
    print(f"\nTraining {model_name}...")
    
    # XGBoost with tuned parameters
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=42,
        eval_metric='logloss'
    )
    
    # RandomForest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    # Voting Ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    ensemble.fit(X_train, y_train)
    
    print(f"✓ {model_name} trained successfully")
    
    return ensemble

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred
    }

def plot_confusion_matrix(cm, model_name, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main pipeline"""
    print("=" * 70)
    print("FINAL IMPROVED MODEL TRAINING - INDIAN STOCK MARKET PREDICTION")
    print("=" * 70)
    
    os.makedirs('models', exist_ok=True)
    
    # Load and engineer features
    df = load_and_engineer_features()
    
    print(f"\nDataset: {len(df)} records")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Model A: Technical features
    print("\n" + "=" * 70)
    print("MODEL A: ENSEMBLE WITH TECHNICAL FEATURES")
    print("=" * 70)
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_tech, features_tech = prepare_datasets(df, include_sentiment=False)
    
    print(f"Training set: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features: {len(features_tech)}")
    
    model_tech = train_ensemble_model(X_train, y_train, X_val, y_val, "Technical Ensemble")
    results_tech = evaluate_model(model_tech, X_test, y_test, "Technical Ensemble")
    
    with open('models/xgb_technical.pkl', 'wb') as f:
        pickle.dump(model_tech, f)
    with open('models/scaler_technical.pkl', 'wb') as f:
        pickle.dump(scaler_tech, f)
    with open('models/features_technical.pkl', 'wb') as f:
        pickle.dump(features_tech, f)
    
    plot_confusion_matrix(results_tech['confusion_matrix'], "Technical Ensemble", 
                         'models/confusion_matrix_technical.png')
    
    # Model B: Hybrid
    print("\n" + "=" * 70)
    print("MODEL B: ENSEMBLE WITH TECHNICAL + SENTIMENT")
    print("=" * 70)
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_hybrid, features_hybrid = prepare_datasets(df, include_sentiment=True)
    
    print(f"Training set: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features: {len(features_hybrid)}")
    
    model_hybrid = train_ensemble_model(X_train, y_train, X_val, y_val, "Hybrid Ensemble")
    results_hybrid = evaluate_model(model_hybrid, X_test, y_test, "Hybrid Ensemble")
    
    with open('models/xgb_hybrid.pkl', 'wb') as f:
        pickle.dump(model_hybrid, f)
    with open('models/scaler_hybrid.pkl', 'wb') as f:
        pickle.dump(scaler_hybrid, f)
    with open('models/features_hybrid.pkl', 'wb') as f:
        pickle.dump(features_hybrid, f)
    
    plot_confusion_matrix(results_hybrid['confusion_matrix'], "Hybrid Ensemble",
                         'models/confusion_matrix_hybrid.png')
    
    # Comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    comparison = pd.DataFrame({
        'Model': ['Technical Ensemble', 'Hybrid Ensemble'],
        'Accuracy': [results_tech['accuracy'], results_hybrid['accuracy']],
        'Precision': [results_tech['precision'], results_hybrid['precision']],
        'Recall': [results_tech['recall'], results_hybrid['recall']],
        'F1-Score': [results_tech['f1_score'], results_hybrid['f1_score']]
    })
    
    print("\n", comparison.to_string(index=False))
    
    improvement = (results_hybrid['accuracy'] - results_tech['accuracy']) * 100
    print(f"\nAccuracy improvement with sentiment: {improvement:+.2f}%")
    
    comparison.to_csv('models/model_comparison.csv', index=False)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
