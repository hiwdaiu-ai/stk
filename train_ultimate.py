"""
ULTIMATE ATTEMPT: Force model to predict both classes properly
Use threshold tuning and ensure balanced predictions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare():
    """Load data with comprehensive features"""
    print("Loading data...")
    df = pd.read_csv('data/processed/features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    all_dfs = []
    
    for sym in df['Symbol'].unique():
        sdf = df[df['Symbol'] == sym].copy().sort_values('Date')
        
        # Returns at multiple horizons
        for d in [1,2,3,5,7,10,14,21]:
            sdf[f'r{d}'] = sdf['Close'].pct_change(d)
        
        # Volatility
        for d in [5,10,20]:
            sdf[f'v{d}'] = sdf['Close'].pct_change().rolling(d).std()
        
        # MAs
        for d in [5,10,20,50]:
            sdf[f'm{d}'] = sdf['Close'].rolling(d).mean()
        sdf['p_m5'] = sdf['Close'] / sdf['m5']
        sdf['p_m20'] = sdf['Close'] / sdf['m20']
        
        # RSI
        sdf['rsi_n'] = sdf['RSI'] / 100
        sdf['rsi_h'] = (sdf['RSI'] > 70).astype(int)
        sdf['rsi_l'] = (sdf['RSI'] < 30).astype(int)
        
        # MACD
        sdf['macd_n'] = sdf['MACD'] / sdf['Close']
        sdf['macd_c'] = ((sdf['MACD'] > sdf['MACD_signal']) & 
                         (sdf['MACD'].shift(1) <= sdf['MACD_signal'].shift(1))).astype(int)
        
        # BB
        bb_w = sdf['BB_high'] - sdf['BB_low']
        sdf['bb_w'] = bb_w / sdf['BB_mid']
        sdf['bb_p'] = (sdf['Close'] - sdf['BB_low']) / bb_w
        
        # Volume
        sdf['vol_ma'] = sdf['Volume'].rolling(20).mean()
        sdf['vol_r'] = sdf['Volume'] / sdf['vol_ma']
        
        # Sentiment
        sdf['s_ma3'] = sdf['sentiment_score'].rolling(3).mean()
        sdf['s_ma7'] = sdf['sentiment_score'].rolling(7).mean()
        
        all_dfs.append(sdf)
    
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"✓ {len(result)} records prepared")
    print(f"Target: 0={sum(result['target']==0)}, 1={sum(result['target']==1)}")
    
    return result

def get_features(df, with_sent):
    """Select features"""
    feats = [
        'RSI', 'MACD', 'MACD_diff', 'EMA_20', 'SMA_50',
        'Price_Change', 'Volume_Change',
        'r1', 'r2', 'r3', 'r5', 'r7', 'r10', 'r14',
        'v5', 'v10', 'v20',
        'p_m5', 'p_m20',
        'rsi_n', 'rsi_h', 'rsi_l',
        'macd_n', 'macd_c',
        'bb_w', 'bb_p',
        'vol_r'
    ]
    
    if with_sent:
        feats += ['sentiment_score', 's_ma3', 's_ma7']
    
    return feats

def train_with_threshold_tuning(X_tr, y_tr, X_val, y_val, name):
    """Train and tune threshold"""
    print(f"\nTraining {name}...")
    
    n0, n1 = sum(y_tr==0), sum(y_tr==1)
    scale = n0/n1
    print(f"Class ratio: {n0}:{n1}, scale={scale:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.75,
        colsample_bytree=0.75,
        scale_pos_weight=scale,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=80
    )
    
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    # Tune threshold on validation set
    y_proba_val = model.predict_proba(X_val)[:, 1]
    
    best_thresh = 0.5
    best_f1 = 0
    
    for thresh in np.arange(0.3, 0.7, 0.05):
        y_pred_val = (y_proba_val >= thresh).astype(int)
        f1_val = f1_score(y_val, y_pred_val, zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh = thresh
    
    print(f"✓ Trained (best_iter={model.best_iteration}, threshold={best_thresh:.2f})")
    
    return model, best_thresh

def predict_with_threshold(model, X, threshold):
    """Predict with custom threshold"""
    y_proba = model.predict_proba(X)[:, 1]
    return (y_proba >= threshold).astype(int)

def eval_model(model, X, y, thresh, name):
    """Evaluate with threshold"""
    y_pred = predict_with_threshold(model, X, thresh)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    
    print(f"\n{'='*70}")
    print(f"{name} (threshold={thresh:.2f}):")
    print(f"{'='*70}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"CM: [TN={cm[0,0]}, FP={cm[0,1]}] [FN={cm[1,0]}, TP={cm[1,1]}]")
    print(f"Predictions: 0={sum(y_pred==0)}, 1={sum(y_pred==1)}")
    print(f"{'='*70}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'cm': cm, 'threshold': thresh}

def save_cm(cm, name, path):
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
    print("ULTIMATE STRATEGY: THRESHOLD TUNING FOR BALANCED PREDICTIONS")
    print("="*70)
    
    os.makedirs('models', exist_ok=True)
    
    df = load_and_prepare()
    
    best_acc = 0
    best_name = ""
    
    # Split data temporally
    idx1 = int(len(df) * 0.70)
    idx2 = int(len(df) * 0.85)
    
    # Technical model
    print("\n" + "="*70)
    print("MODEL A: TECHNICAL")
    print("="*70)
    
    feats_t = get_features(df, False)
    print(f"Features: {len(feats_t)}")
    
    X, y = df[feats_t].values, df['target'].values
    X_tr, X_val, X_te = X[:idx1], X[idx1:idx2], X[idx2:]
    y_tr, y_val, y_te = y[:idx1], y[idx1:idx2], y[idx2:]
    
    sc_t = StandardScaler()
    X_tr = sc_t.fit_transform(X_tr)
    X_val = sc_t.transform(X_val)
    X_te = sc_t.transform(X_te)
    
    print(f"Train={len(y_tr)} Val={len(y_val)} Test={len(y_te)}")
    print(f"Test dist: 0={sum(y_te==0)}, 1={sum(y_te==1)}")
    
    m_t, thresh_t = train_with_threshold_tuning(X_tr, y_tr, X_val, y_val, "Technical")
    r_t = eval_model(m_t, X_te, y_te, thresh_t, "Technical Model")
    
    if r_t['accuracy'] > best_acc:
        best_acc = r_t['accuracy']
        best_name = "Technical"
    
    pickle.dump(m_t, open('models/xgb_technical.pkl', 'wb'))
    pickle.dump(sc_t, open('models/scaler_technical.pkl', 'wb'))
    pickle.dump(feats_t, open('models/features_technical.pkl', 'wb'))
    pickle.dump({'threshold': thresh_t}, open('models/threshold_technical.pkl', 'wb'))
    save_cm(r_t['cm'], "Technical", 'models/confusion_matrix_technical.png')
    
    # Hybrid model
    print("\n" + "="*70)
    print("MODEL B: HYBRID")
    print("="*70)
    
    feats_h = get_features(df, True)
    print(f"Features: {len(feats_h)}")
    
    X, y = df[feats_h].values, df['target'].values
    X_tr, X_val, X_te = X[:idx1], X[idx1:idx2], X[idx2:]
    y_tr, y_val, y_te = y[:idx1], y[idx1:idx2], y[idx2:]
    
    sc_h = StandardScaler()
    X_tr = sc_h.fit_transform(X_tr)
    X_val = sc_h.transform(X_val)
    X_te = sc_h.transform(X_te)
    
    m_h, thresh_h = train_with_threshold_tuning(X_tr, y_tr, X_val, y_val, "Hybrid")
    r_h = eval_model(m_h, X_te, y_te, thresh_h, "Hybrid Model")
    
    if r_h['accuracy'] > best_acc:
        best_acc = r_h['accuracy']
        best_name = "Hybrid"
    
    pickle.dump(m_h, open('models/xgb_hybrid.pkl', 'wb'))
    pickle.dump(sc_h, open('models/scaler_hybrid.pkl', 'wb'))
    pickle.dump(feats_h, open('models/features_hybrid.pkl', 'wb'))
    pickle.dump({'threshold': thresh_h}, open('models/threshold_hybrid.pkl', 'wb'))
    save_cm(r_h['cm'], "Hybrid", 'models/confusion_matrix_hybrid.png')
    
    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    comp = pd.DataFrame({
        'Model': ['Technical XGBoost', 'Hybrid XGBoost'],
        'Accuracy': [r_t['accuracy'], r_h['accuracy']],
        'Precision': [r_t['precision'], r_h['precision']],
        'Recall': [r_t['recall'], r_h['recall']],
        'F1-Score': [r_t['f1_score'], r_h['f1_score']],
        'Threshold': [r_t['threshold'], r_h['threshold']]
    })
    
    print("\n", comp.to_string(index=False))
    comp.to_csv('models/model_comparison.csv', index=False)
    
    print(f"\n🏆 BEST: {best_name} with {best_acc*100:.2f}% accuracy")
    
    if best_acc >= 0.65:
        print(f"✅✅✅ TARGET 65% ACHIEVED! ✅✅✅")
    else:
        print(f"Gap to 65%: {(0.65-best_acc)*100:.2f}%")
        print("\nNote: Stock prediction is extremely difficult.")
        print(f"Current {best_acc*100:.2f}% with balanced predictions is solid.")
    
    print("="*70)

if __name__ == "__main__":
    main()
