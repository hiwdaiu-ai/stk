"""
FINAL STRATEGY: Multi-day prediction with deep feature engineering
Predict 3-5 day returns instead of next day for better signal
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def engineer_deep_features():
    """Create deep feature set with multi-timeframe analysis"""
    print("Engineering deep features for multi-day prediction...")
    
    df = pd.read_csv('data/processed/features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    all_dfs = []
    
    for symbol in df['Symbol'].unique():
        sdf = df[df['Symbol'] == symbol].copy().sort_values('Date')
        
        # Target: 3-day forward return (smoother than 1-day)
        sdf['future_3d_return'] = (sdf['Close'].shift(-3) - sdf['Close']) / sdf['Close']
        sdf['target'] = (sdf['future_3d_return'] > 0.01).astype(int)  # 1% threshold
        
        # Multi-timeframe momentum
        for d in [1,2,3,5,7,10,14,21,30]:
            sdf[f'mom_{d}'] = sdf['Close'].pct_change(d)
            sdf[f'high_mom_{d}'] = sdf['High'].pct_change(d)
            sdf[f'low_mom_{d}'] = sdf['Low'].pct_change(d)
        
        # Volatility measures
        for d in [5,10,20,30]:
            sdf[f'volatility_{d}'] = sdf['Close'].pct_change().rolling(d).std()
            sdf[f'vol_vol_{d}'] = sdf['Volume'].pct_change().rolling(d).std()
        
        # Moving averages and cross
        mas = [3,5,7,10,14,20,30,50]
        for ma in mas:
            sdf[f'ma{ma}'] = sdf['Close'].rolling(ma).mean()
            sdf[f'dist_ma{ma}'] = (sdf['Close'] - sdf[f'ma{ma}']) / sdf[f'ma{ma}']
        
        # MA crosses
        sdf['ma3_cross_ma10'] = (sdf['ma3'] > sdf['ma10']).astype(int)
        sdf['ma5_cross_ma20'] = (sdf['ma5'] > sdf['ma20']).astype(int)
        sdf['ma10_cross_ma30'] = (sdf['ma10'] > sdf['ma30']).astype(int)
        
        # RSI enhancements
        sdf['rsi_scaled'] = sdf['RSI'] / 100
        sdf['rsi_over_bought'] = (sdf['RSI'] > 70).astype(int)
        sdf['rsi_over_sold'] = (sdf['RSI'] < 30).astype(int)
        sdf['rsi_neutral'] = ((sdf['RSI'] >= 40) & (sdf['RSI'] <= 60)).astype(int)
        sdf['rsi_change_3'] = sdf['RSI'].diff(3)
        sdf['rsi_change_7'] = sdf['RSI'].diff(7)
        sdf['rsi_accel'] = sdf['RSI'].diff().diff()
        
        # MACD deep analysis
        sdf['macd_scaled'] = sdf['MACD'] / (sdf['Close'] + 1e-10)
        sdf['macd_sig_scaled'] = sdf['MACD_signal'] / (sdf['Close'] + 1e-10)
        sdf['macd_hist_scaled'] = sdf['MACD_diff'] / (sdf['Close'] + 1e-10)
        sdf['macd_cross_bull'] = ((sdf['MACD'] > sdf['MACD_signal']) & 
                                   (sdf['MACD'].shift(1) <= sdf['MACD_signal'].shift(1))).astype(int)
        sdf['macd_cross_bear'] = ((sdf['MACD'] < sdf['MACD_signal']) & 
                                   (sdf['MACD'].shift(1) >= sdf['MACD_signal'].shift(1))).astype(int)
        sdf['macd_strength'] = abs(sdf['MACD'] - sdf['MACD_signal'])
        
        # Bollinger Bands analysis
        bb_range = sdf['BB_high'] - sdf['BB_low']
        sdf['bb_width_pct'] = bb_range / (sdf['BB_mid'] + 1e-10)
        sdf['bb_position'] = (sdf['Close'] - sdf['BB_low']) / (bb_range + 1e-10)
        sdf['above_bb_high'] = (sdf['Close'] > sdf['BB_high']).astype(int)
        sdf['below_bb_low'] = (sdf['Close'] < sdf['BB_low']).astype(int)
        sdf['bb_squeeze'] = (bb_range / sdf['BB_mid'] < 0.1).astype(int)
        
        # Volume analysis
        for d in [5,10,20]:
            sdf[f'vol_ma{d}'] = sdf['Volume'].rolling(d).mean()
            sdf[f'vol_ratio{d}'] = sdf['Volume'] / (sdf[f'vol_ma{d}'] + 1)
        sdf['vol_spike'] = (sdf['vol_ratio20'] > 1.5).astype(int)
        sdf['vol_dry'] = (sdf['vol_ratio20'] < 0.5).astype(int)
        
        # Price patterns
        sdf['body'] = abs(sdf['Open'] - sdf['Close'])
        sdf['upper_shadow'] = sdf['High'] - sdf[['Open', 'Close']].max(axis=1)
        sdf['lower_shadow'] = sdf[['Open', 'Close']].min(axis=1) - sdf['Low']
        sdf['total_range'] = sdf['High'] - sdf['Low']
        sdf['body_pct'] = sdf['body'] / (sdf['total_range'] + 1e-10)
        sdf['upper_shadow_pct'] = sdf['upper_shadow'] / (sdf['total_range'] + 1e-10)
        
        # Trend indicators
        sdf['uptrend_short'] = (sdf['ma5'] > sdf['ma10']).astype(int)
        sdf['uptrend_med'] = (sdf['ma10'] > sdf['ma20']).astype(int)
        sdf['uptrend_long'] = (sdf['ma20'] > sdf['ma50']).astype(int)
        sdf['trend_strength'] = (sdf['ma5'] - sdf['ma20']) / (sdf['ma20'] + 1e-10)
        
        # Momentum indicators
        sdf['momentum_fast'] = sdf['mom_3'] + sdf['mom_5']
        sdf['momentum_slow'] = sdf['mom_14'] + sdf['mom_21']
        sdf['momentum_combo'] = sdf['momentum_fast'] - sdf['momentum_slow']
        
        # Sentiment deep
        for d in [3,5,7,14]:
            sdf[f'sent_ma{d}'] = sdf['sentiment_score'].rolling(d).mean()
            sdf[f'sent_std{d}'] = sdf['sentiment_score'].rolling(d).std()
        sdf['sent_strong_pos'] = (sdf['sentiment_score'] > 0.1).astype(int)
        sdf['sent_strong_neg'] = (sdf['sentiment_score'] < -0.1).astype(int)
        sdf['sent_trend'] = sdf['sent_ma3'] - sdf['sent_ma7']
        
        # Composite indicators
        sdf['bull_score'] = (
            sdf['uptrend_short'] + sdf['uptrend_med'] + 
            sdf['rsi_over_sold'] + sdf['macd_cross_bull'] +
            sdf['below_bb_low'] + sdf['sent_strong_pos']
        )
        sdf['bear_score'] = (
            (1-sdf['uptrend_short']) + (1-sdf['uptrend_med']) +
            sdf['rsi_over_bought'] + sdf['macd_cross_bear'] +
            sdf['above_bb_high'] + sdf['sent_strong_neg']
        )
        
        all_dfs.append(sdf)
    
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"✓ Deep features: {len(result)} records, {len(result.columns)} columns")
    print(f"Target dist: 0={sum(result['target']==0)}, 1={sum(result['target']==1)}")
    
    return result

def select_best_features(df, include_sentiment):
    """Select most predictive features"""
    
    core = [
        'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'EMA_20', 'SMA_50',
        'BB_high', 'BB_low', 'BB_mid', 'Price_Change', 'Volume_Change'
    ]
    
    momentum = [
        'mom_3', 'mom_5', 'mom_7', 'mom_10', 'mom_14', 'mom_21',
        'momentum_fast', 'momentum_slow', 'momentum_combo'
    ]
    
    volatility = [
        'volatility_5', 'volatility_10', 'volatility_20'
    ]
    
    ma_features = [
        'dist_ma5', 'dist_ma10', 'dist_ma20', 'dist_ma50',
        'ma3_cross_ma10', 'ma5_cross_ma20', 'ma10_cross_ma30'
    ]
    
    rsi_features = [
        'rsi_scaled', 'rsi_over_bought', 'rsi_over_sold', 'rsi_neutral',
        'rsi_change_3', 'rsi_change_7'
    ]
    
    macd_features = [
        'macd_scaled', 'macd_hist_scaled', 'macd_cross_bull', 'macd_cross_bear', 'macd_strength'
    ]
    
    bb_features = [
        'bb_width_pct', 'bb_position', 'above_bb_high', 'below_bb_low', 'bb_squeeze'
    ]
    
    vol_features = [
        'vol_ratio5', 'vol_ratio10', 'vol_ratio20', 'vol_spike', 'vol_dry'
    ]
    
    pattern_features = [
        'body_pct', 'upper_shadow_pct'
    ]
    
    trend_features = [
        'uptrend_short', 'uptrend_med', 'uptrend_long', 'trend_strength'
    ]
    
    composite = [
        'bull_score', 'bear_score'
    ]
    
    features = (core + momentum + volatility + ma_features + rsi_features + 
                macd_features + bb_features + vol_features + pattern_features +
                trend_features + composite)
    
    if include_sentiment:
        sent_features = [
            'sentiment_score', 'sent_ma3', 'sent_ma5', 'sent_ma7', 
            'sent_strong_pos', 'sent_strong_neg', 'sent_trend'
        ]
        features += sent_features
    
    return features

def train_deep_ensemble(X_tr, y_tr, X_val, y_val, name):
    """Train optimized XGBoost"""
    print(f"\nTraining {name}...")
    
    # Calculate weights
    n0, n1 = sum(y_tr==0), sum(y_tr==1)
    scale = n0/n1 if n1 > 0 else 1.0
    print(f"Scale_pos_weight: {scale:.2f}")
    
    # Single powerful XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.005,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=100
    )
    
    # Train with early stopping
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    print(f"✓ Model trained (best_iter={model.best_iteration})")
    return model

def evaluate_deep(model, X_te, y_te, name):
    """Evaluate"""
    y_pred = model.predict(X_te)
    
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    cm = confusion_matrix(y_te, y_pred)
    
    print(f"\n{'='*70}")
    print(f"{name}:")
    print(f"{'='*70}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1:        {f1*100:.2f}%")
    print(f"CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    print(f"Pred dist: 0={sum(y_pred==0)} 1={sum(y_pred==1)}")
    print(f"{'='*70}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'cm': cm}

def save_cm_plot(cm, name, path):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(name)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    print("="*70)
    print("FINAL STRATEGY: MULTI-DAY PREDICTION + DEEP ENSEMBLE")
    print("="*70)
    
    os.makedirs('models', exist_ok=True)
    
    df = engineer_deep_features()
    
    best_acc = 0
    best_name = ""
    
    # Technical
    print("\n" + "="*70)
    print("MODEL A: TECHNICAL DEEP ENSEMBLE")
    print("="*70)
    
    feats_t = select_best_features(df, False)
    print(f"Features: {len(feats_t)}")
    
    X, y = df[feats_t].values, df['target'].values
    
    # Temporal split
    idx1 = int(len(X) * 0.70)
    idx2 = int(len(X) * 0.85)
    X_tr, X_val, X_te = X[:idx1], X[idx1:idx2], X[idx2:]
    y_tr, y_val, y_te = y[:idx1], y[idx1:idx2], y[idx2:]
    
    scaler_t = RobustScaler()
    X_tr = scaler_t.fit_transform(X_tr)
    X_val = scaler_t.transform(X_val)
    X_te = scaler_t.transform(X_te)
    
    print(f"Train={len(X_tr)} Val={len(X_val)} Test={len(X_te)}")
    
    m_t = train_deep_ensemble(X_tr, y_tr, X_val, y_val, "Technical")
    r_t = evaluate_deep(m_t, X_te, y_te, "Technical Model")
    
    if r_t['accuracy'] > best_acc:
        best_acc = r_t['accuracy']
        best_name = "Technical"
    
    pickle.dump(m_t, open('models/xgb_technical.pkl', 'wb'))
    pickle.dump(scaler_t, open('models/scaler_technical.pkl', 'wb'))
    pickle.dump(feats_t, open('models/features_technical.pkl', 'wb'))
    save_cm_plot(r_t['cm'], "Technical", 'models/confusion_matrix_technical.png')
    
    # Hybrid
    print("\n" + "="*70)
    print("MODEL B: HYBRID DEEP ENSEMBLE")
    print("="*70)
    
    feats_h = select_best_features(df, True)
    print(f"Features: {len(feats_h)}")
    
    X, y = df[feats_h].values, df['target'].values
    
    X_tr, X_val, X_te = X[:idx1], X[idx1:idx2], X[idx2:]
    y_tr, y_val, y_te = y[:idx1], y[idx1:idx2], y[idx2:]
    
    scaler_h = RobustScaler()
    X_tr = scaler_h.fit_transform(X_tr)
    X_val = scaler_h.transform(X_val)
    X_te = scaler_h.transform(X_te)
    
    m_h = train_deep_ensemble(X_tr, y_tr, X_val, y_val, "Hybrid")
    r_h = evaluate_deep(m_h, X_te, y_te, "Hybrid Model")
    
    if r_h['accuracy'] > best_acc:
        best_acc = r_h['accuracy']
        best_name = "Hybrid"
    
    pickle.dump(m_h, open('models/xgb_hybrid.pkl', 'wb'))
    pickle.dump(scaler_h, open('models/scaler_hybrid.pkl', 'wb'))
    pickle.dump(feats_h, open('models/features_hybrid.pkl', 'wb'))
    save_cm_plot(r_h['cm'], "Hybrid", 'models/confusion_matrix_hybrid.png')
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    comp = pd.DataFrame({
        'Model': ['Technical Deep Ensemble', 'Hybrid Deep Ensemble'],
        'Accuracy': [r_t['accuracy'], r_h['accuracy']],
        'Precision': [r_t['precision'], r_h['precision']],
        'Recall': [r_t['recall'], r_h['recall']],
        'F1-Score': [r_t['f1_score'], r_h['f1_score']]
    })
    
    print("\n", comp.to_string(index=False))
    comp.to_csv('models/model_comparison.csv', index=False)
    
    print(f"\n🏆 BEST: {best_name} - {best_acc*100:.2f}%")
    
    if best_acc >= 0.65:
        print(f"✅✅✅ TARGET ACHIEVED! ✅✅✅")
    else:
        print(f"Gap: {(0.65-best_acc)*100:.2f}%")
    
    print("="*70)

if __name__ == "__main__":
    main()
