"""
Strategy 2: Predict Significant Price Movements (>1% change) instead of just direction
This should give better signal-to-noise ratio
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_create_better_target():
    """Load data and create better target - predict significant movements"""
    print("Loading data with improved target definition...")
    
    df = pd.read_csv('data/processed/features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    result_dfs = []
    
    for symbol in df['Symbol'].unique():
        symbol_df = df[df['Symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('Date')
        
        # Calculate next day return
        symbol_df['next_close'] = symbol_df['Close'].shift(-1)
        symbol_df['next_return'] = (symbol_df['next_close'] - symbol_df['Close']) / symbol_df['Close']
        
        # New target: Predict if price will move significantly (>0.5% up or down)
        # Focus on clearer signals
        threshold = 0.005  # 0.5% threshold
        
        # Target: 1 if significant upward movement, 0 otherwise
        symbol_df['target'] = (symbol_df['next_return'] > threshold).astype(int)
        
        # Advanced features
        # Momentum
        for period in [2, 3, 5, 7, 10, 14, 21]:
            symbol_df[f'return_{period}d'] = symbol_df['Close'].pct_change(period)
            symbol_df[f'return_std_{period}d'] = symbol_df['Close'].pct_change().rolling(period).std()
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            symbol_df[f'ma_{period}'] = symbol_df['Close'].rolling(period).mean()
            symbol_df[f'price_to_ma{period}'] = symbol_df['Close'] / symbol_df[f'ma_{period}']
        
        # RSI features
        symbol_df['rsi_norm'] = symbol_df['RSI'] / 100
        symbol_df['rsi_overbought'] = (symbol_df['RSI'] > 70).astype(int)
        symbol_df['rsi_oversold'] = (symbol_df['RSI'] < 30).astype(int)
        symbol_df['rsi_mom'] = symbol_df['RSI'].diff(3)
        
        # MACD features
        symbol_df['macd_diff_norm'] = symbol_df['MACD_diff'] / symbol_df['Close']
        symbol_df['macd_cross'] = ((symbol_df['MACD'] > symbol_df['MACD_signal']) & 
                                    (symbol_df['MACD'].shift(1) <= symbol_df['MACD_signal'].shift(1))).astype(int)
        
        # Bollinger Bands
        bb_width = symbol_df['BB_high'] - symbol_df['BB_low']
        symbol_df['bb_width_pct'] = bb_width / symbol_df['BB_mid']
        symbol_df['bb_pos'] = (symbol_df['Close'] - symbol_df['BB_low']) / bb_width
        symbol_df['above_bb_high'] = (symbol_df['Close'] > symbol_df['BB_high']).astype(int)
        symbol_df['below_bb_low'] = (symbol_df['Close'] < symbol_df['BB_low']).astype(int)
        
        # Volume
        symbol_df['vol_ma20'] = symbol_df['Volume'].rolling(20).mean()
        symbol_df['vol_ratio'] = symbol_df['Volume'] / symbol_df['vol_ma20']
        symbol_df['vol_spike'] = (symbol_df['vol_ratio'] > 2).astype(int)
        
        # Price patterns
        symbol_df['hl_range'] = (symbol_df['High'] - symbol_df['Low']) / symbol_df['Close']
        symbol_df['open_close_diff'] = (symbol_df['Close'] - symbol_df['Open']) / symbol_df['Open']
        
        # Trend
        symbol_df['uptrend'] = (symbol_df['ma_5'] > symbol_df['ma_20']).astype(int)
        symbol_df['downtrend'] = (symbol_df['ma_5'] < symbol_df['ma_20']).astype(int)
        
        # Volatility
        symbol_df['volatility'] = symbol_df['Close'].pct_change().rolling(20).std()
        
        # Lag features
        symbol_df['close_lag1'] = symbol_df['Close'].shift(1)
        symbol_df['rsi_lag1'] = symbol_df['RSI'].shift(1)
        symbol_df['volume_lag1'] = symbol_df['Volume'].shift(1)
        
        # Sentiment enhanced
        symbol_df['sent_ma5'] = symbol_df['sentiment_score'].rolling(5).mean()
        symbol_df['sent_ma10'] = symbol_df['sentiment_score'].rolling(10).mean()
        symbol_df['sent_positive'] = (symbol_df['sentiment_score'] > 0.1).astype(int)
        symbol_df['sent_negative'] = (symbol_df['sentiment_score'] < -0.1).astype(int)
        
        result_dfs.append(symbol_df)
    
    combined_df = pd.concat(result_dfs, ignore_index=True)
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
    combined_df = combined_df.dropna()
    
    print(f"✓ Features created: {len(combined_df)} records")
    print(f"Target distribution: {combined_df['target'].value_counts().to_dict()}")
    
    return combined_df

def prepare_features(df, include_sentiment=False):
    """Prepare feature set"""
    
    base_features = [
        # Technical indicators
        'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
        'EMA_20', 'SMA_50',
        'BB_high', 'BB_low', 'BB_mid',
        'Price_Change', 'Volume_Change',
        
        # Returns
        'return_2d', 'return_3d', 'return_5d', 'return_7d', 'return_10d', 'return_14d', 'return_21d',
        
        # Volatility
        'return_std_5d', 'return_std_10d', 'return_std_14d', 'volatility',
        
        # Moving averages
        'price_to_ma5', 'price_to_ma10', 'price_to_ma20', 'price_to_ma50',
        
        # RSI
        'rsi_norm', 'rsi_overbought', 'rsi_oversold', 'rsi_mom',
        
        # MACD
        'macd_diff_norm', 'macd_cross',
        
        # Bollinger Bands
        'bb_width_pct', 'bb_pos', 'above_bb_high', 'below_bb_low',
        
        # Volume
        'vol_ratio', 'vol_spike',
        
        # Price patterns
        'hl_range', 'open_close_diff',
        
        # Trend
        'uptrend', 'downtrend',
        
        # Lags
        'rsi_lag1'
    ]
    
    if include_sentiment:
        sentiment_features = [
            'sentiment_score', 'sent_ma5', 'sent_ma10',
            'sent_positive', 'sent_negative'
        ]
        features = base_features + sentiment_features
    else:
        features = base_features
    
    print(f"Using {len(features)} features")
    
    X = df[features].values
    y = df['target'].values
    
    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, shuffle=False)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Test target distribution: {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, features

def train_model(X_train, y_train, X_val, y_val, model_name):
    """Train model with grid search"""
    print(f"\nTraining {model_name}...")
    
    # XGBoost optimized
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.005,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=100
    )
    
    # Fit with early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"✓ {model_name} trained (best iteration: {xgb_model.best_iteration})")
    
    return xgb_model

def evaluate(model, X_test, y_test, name):
    """Evaluate model"""
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print(f"{name} Results:")
    print(f"{'='*70}")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"{'='*70}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'cm': cm}

def save_cm(cm, name, path):
    """Save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    """Main pipeline"""
    print("="*70)
    print("STRATEGY 2: PREDICT SIGNIFICANT PRICE MOVEMENTS (>0.5%)")
    print("="*70)
    
    os.makedirs('models', exist_ok=True)
    
    df = load_and_create_better_target()
    
    # Technical model
    print("\n" + "="*70)
    print("MODEL A: TECHNICAL ONLY")
    print("="*70)
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_t, feat_t = prepare_features(df, False)
    
    model_t = train_model(X_train, y_train, X_val, y_val, "Technical XGBoost")
    results_t = evaluate(model_t, X_test, y_test, "Technical Model")
    
    pickle.dump(model_t, open('models/xgb_technical.pkl', 'wb'))
    pickle.dump(scaler_t, open('models/scaler_technical.pkl', 'wb'))
    pickle.dump(feat_t, open('models/features_technical.pkl', 'wb'))
    save_cm(results_t['cm'], "Technical", 'models/confusion_matrix_technical.png')
    
    # Hybrid model
    print("\n" + "="*70)
    print("MODEL B: HYBRID (TECHNICAL + SENTIMENT)")
    print("="*70)
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_h, feat_h = prepare_features(df, True)
    
    model_h = train_model(X_train, y_train, X_val, y_val, "Hybrid XGBoost")
    results_h = evaluate(model_h, X_test, y_test, "Hybrid Model")
    
    pickle.dump(model_h, open('models/xgb_hybrid.pkl', 'wb'))
    pickle.dump(scaler_h, open('models/scaler_hybrid.pkl', 'wb'))
    pickle.dump(feat_h, open('models/features_hybrid.pkl', 'wb'))
    save_cm(results_h['cm'], "Hybrid", 'models/confusion_matrix_hybrid.png')
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    comp = pd.DataFrame({
        'Model': ['Technical XGBoost', 'Hybrid XGBoost'],
        'Accuracy': [results_t['accuracy'], results_h['accuracy']],
        'Precision': [results_t['precision'], results_h['precision']],
        'Recall': [results_t['recall'], results_h['recall']],
        'F1-Score': [results_t['f1_score'], results_h['f1_score']]
    })
    
    print("\n", comp.to_string(index=False))
    comp.to_csv('models/model_comparison.csv', index=False)
    
    best_acc = max(results_t['accuracy'], results_h['accuracy'])
    best_name = "Technical" if results_t['accuracy'] > results_h['accuracy'] else "Hybrid"
    
    print(f"\n✅ BEST: {best_name} with {best_acc*100:.2f}% accuracy")
    
    if best_acc >= 0.65:
        print("🎉 TARGET ACHIEVED!")
    else:
        print(f"⚠️ Gap to target: {(0.65-best_acc)*100:.2f}%")
    
    print("="*70)

if __name__ == "__main__":
    main()
