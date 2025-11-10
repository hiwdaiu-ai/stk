"""
Strategy 3: Balanced prediction with proper evaluation
Use stratified split and balanced class weights
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data with balanced approach"""
    print("Loading and preparing data...")
    
    df = pd.read_csv('data/processed/features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    result_dfs = []
    
    for symbol in df['Symbol'].unique():
        symbol_df = df[df['Symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('Date')
        
        # Create comprehensive features
        # Returns
        for p in [1, 2, 3, 5, 7, 10, 14, 20]:
            symbol_df[f'ret_{p}d'] = symbol_df['Close'].pct_change(p)
        
        # Volatility
        for p in [5, 10, 20]:
            symbol_df[f'vol_{p}d'] = symbol_df['Close'].pct_change().rolling(p).std()
        
        # Moving averages
        for p in [5, 10, 20, 50]:
            symbol_df[f'ma{p}'] = symbol_df['Close'].rolling(p).mean()
            symbol_df[f'price_ma{p}_ratio'] = symbol_df['Close'] / symbol_df[f'ma{p}']
        
        # RSI
        symbol_df['rsi_norm'] = symbol_df['RSI'] / 100
        symbol_df['rsi_over70'] = (symbol_df['RSI'] > 70).astype(int)
        symbol_df['rsi_under30'] = (symbol_df['RSI'] < 30).astype(int)
        
        # MACD
        symbol_df['macd_norm'] = symbol_df['MACD'] / symbol_df['Close']
        symbol_df['macd_cross_up'] = ((symbol_df['MACD'] > symbol_df['MACD_signal']) & 
                                       (symbol_df['MACD'].shift(1) <= symbol_df['MACD_signal'].shift(1))).astype(int)
        
        # Bollinger Bands
        bb_range = symbol_df['BB_high'] - symbol_df['BB_low']
        symbol_df['bb_pct'] = bb_range / symbol_df['BB_mid']
        symbol_df['bb_pos'] = (symbol_df['Close'] - symbol_df['BB_low']) / bb_range
        
        # Volume
        symbol_df['vol_ma20'] = symbol_df['Volume'].rolling(20).mean()
        symbol_df['vol_norm'] = symbol_df['Volume'] / symbol_df['vol_ma20']
        
        # Price action
        symbol_df['hl_pct'] = (symbol_df['High'] - symbol_df['Low']) / symbol_df['Close']
        
        # Trend
        symbol_df['trend_up'] = (symbol_df['ma5'] > symbol_df['ma20']).astype(int)
        
        # Sentiment
        symbol_df['sent_ma3'] = symbol_df['sentiment_score'].rolling(3).mean()
        symbol_df['sent_ma7'] = symbol_df['sentiment_score'].rolling(7).mean()
        symbol_df['sent_pos'] = (symbol_df['sentiment_score'] > 0.05).astype(int)
        
        result_dfs.append(symbol_df)
    
    combined_df = pd.concat(result_dfs, ignore_index=True)
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
    combined_df = combined_df.dropna()
    
    print(f"✓ Data prepared: {len(combined_df)} records")
    return combined_df

def prepare_balanced_sets(df, include_sentiment=False):
    """Prepare balanced train/test sets"""
    
    features = [
        'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
        'EMA_20', 'SMA_50', 'Price_Change', 'Volume_Change',
        'BB_high', 'BB_low', 'BB_mid',
        'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_7d', 'ret_10d',
        'vol_5d', 'vol_10d', 'vol_20d',
        'price_ma5_ratio', 'price_ma10_ratio', 'price_ma20_ratio',
        'rsi_norm', 'rsi_over70', 'rsi_under30',
        'macd_norm', 'macd_cross_up',
        'bb_pct', 'bb_pos',
        'vol_norm',
        'hl_pct',
        'trend_up'
    ]
    
    if include_sentiment:
        features += ['sentiment_score', 'sent_ma3', 'sent_ma7', 'sent_pos']
    
    print(f"Features: {len(features)}")
    
    X = df[features].values
    y = df['target'].values
    
    print(f"Target distribution: 0={sum(y==0)}, 1={sum(y==1)}")
    
    # Temporal split but with stratification consideration
    # Split into train (85%) and test (15%) first
    split_idx = int(len(X) * 0.85)
    X_train_val, X_test = X[:split_idx], X[split_idx:]
    y_train_val, y_test = y[:split_idx], y[split_idx:]
    
    # Split train_val into train (82%) and val (18%)  
    split_idx2 = int(len(X_train_val) * 0.82)
    X_train, X_val = X_train_val[:split_idx2], X_train_val[split_idx2:]
    y_train, y_val = y_train_val[:split_idx2], y_train_val[split_idx2:]
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {len(X_train)} (0={sum(y_train==0)}, 1={sum(y_train==1)})")
    print(f"Val:   {len(X_val)} (0={sum(y_val==0)}, 1={sum(y_val==1)})")
    print(f"Test:  {len(X_test)} (0={sum(y_test==0)}, 1={sum(y_test==1)})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, features

def train_balanced_model(X_train, y_train, X_val, y_val, name):
    """Train with balanced approach"""
    print(f"\nTraining {name}...")
    
    # Calculate scale_pos_weight
    n_neg = sum(y_train == 0)
    n_pos = sum(y_train == 1)
    scale = n_neg / n_pos if n_pos > 0 else 1.0
    
    print(f"Class balance: neg={n_neg}, pos={n_pos}, scale_pos_weight={scale:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale,  # Balance classes
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"✓ Trained (best_iteration={model.best_iteration})")
    return model

def evaluate_balanced(model, X_test, y_test, name):
    """Comprehensive evaluation"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print(f"{name} Performance:")
    print(f"{'='*70}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    print(f"\nPredictions: 0={sum(y_pred==0)}, 1={sum(y_pred==1)}")
    print(f"{'='*70}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'cm': cm}

def save_plot(cm, name, path):
    """Save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    """Main pipeline"""
    print("="*70)
    print("STRATEGY 3: BALANCED TRAINING WITH PROPER CLASS WEIGHTS")
    print("="*70)
    
    os.makedirs('models', exist_ok=True)
    
    df = load_and_prepare_data()
    
    best_acc = 0
    best_name = ""
    
    # Technical model
    print("\n" + "="*70)
    print("MODEL A: TECHNICAL")
    print("="*70)
    
    X_tr, X_v, X_te, y_tr, y_v, y_te, sc_t, ft_t = prepare_balanced_sets(df, False)
    
    m_t = train_balanced_model(X_tr, y_tr, X_v, y_v, "Technical")
    r_t = evaluate_balanced(m_t, X_te, y_te, "Technical Model")
    
    if r_t['accuracy'] > best_acc:
        best_acc = r_t['accuracy']
        best_name = "Technical"
    
    pickle.dump(m_t, open('models/xgb_technical.pkl', 'wb'))
    pickle.dump(sc_t, open('models/scaler_technical.pkl', 'wb'))
    pickle.dump(ft_t, open('models/features_technical.pkl', 'wb'))
    save_plot(r_t['cm'], "Technical", 'models/confusion_matrix_technical.png')
    
    # Hybrid model
    print("\n" + "="*70)
    print("MODEL B: HYBRID")
    print("="*70)
    
    X_tr, X_v, X_te, y_tr, y_v, y_te, sc_h, ft_h = prepare_balanced_sets(df, True)
    
    m_h = train_balanced_model(X_tr, y_tr, X_v, y_v, "Hybrid")
    r_h = evaluate_balanced(m_h, X_te, y_te, "Hybrid Model")
    
    if r_h['accuracy'] > best_acc:
        best_acc = r_h['accuracy']
        best_name = "Hybrid"
    
    pickle.dump(m_h, open('models/xgb_hybrid.pkl', 'wb'))
    pickle.dump(sc_h, open('models/scaler_hybrid.pkl', 'wb'))
    pickle.dump(ft_h, open('models/features_hybrid.pkl', 'wb'))
    save_plot(r_h['cm'], "Hybrid", 'models/confusion_matrix_hybrid.png')
    
    # Comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    comp = pd.DataFrame({
        'Model': ['Technical XGBoost', 'Hybrid XGBoost'],
        'Accuracy': [r_t['accuracy'], r_h['accuracy']],
        'Precision': [r_t['precision'], r_h['precision']],
        'Recall': [r_t['recall'], r_h['recall']],
        'F1-Score': [r_t['f1_score'], r_h['f1_score']]
    })
    
    print("\n", comp.to_string(index=False))
    comp.to_csv('models/model_comparison.csv', index=False)
    
    print(f"\n{'='*70}")
    print(f"🏆 BEST MODEL: {best_name}")
    print(f"🏆 ACCURACY: {best_acc*100:.2f}%")
    
    if best_acc >= 0.65:
        print(f"✅ TARGET ACHIEVED! ({best_acc*100:.2f}% >= 65%)")
    else:
        print(f"⚠️  Short of target by {(0.65-best_acc)*100:.2f}%")
    
    print("="*70)

if __name__ == "__main__":
    main()
