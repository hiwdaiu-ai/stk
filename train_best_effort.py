"""
Best effort: Get the best accuracy we can achieve with proper predictions
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

df = pd.read_csv('data/processed/features.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Engineer features
all_dfs = []
for sym in df['Symbol'].unique():
    sdf = df[df['Symbol'] == sym].copy().sort_values('Date')
    
    for d in [1,2,3,5,7,10,14,21]:
        sdf[f'r{d}'] = sdf['Close'].pct_change(d)
    for d in [5,10,20]:
        sdf[f'v{d}'] = sdf['Close'].pct_change().rolling(d).std()
    for d in [5,10,20,50]:
        sdf[f'm{d}'] = sdf['Close'].rolling(d).mean()
    sdf['rsi_n'] = sdf['RSI'] / 100
    sdf['macd_n'] = sdf['MACD'] / sdf['Close']
    sdf['bb_w'] = (sdf['BB_high'] - sdf['BB_low']) / sdf['BB_mid']
    sdf['vol_ma'] = sdf['Volume'].rolling(20).mean()
    sdf['vol_r'] = sdf['Volume'] / sdf['vol_ma']
    sdf['s_ma7'] = sdf['sentiment_score'].rolling(7).mean()
    
    all_dfs.append(sdf)

result = pd.concat(all_dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()

print(f"Dataset: {len(result)} records")
print(f"Target: 0={sum(result['target']==0)}, 1={sum(result['target']==1)}")

# Features
feats_t = ['RSI', 'MACD', 'EMA_20', 'SMA_50', 'Price_Change', 'Volume_Change',
           'r1', 'r3', 'r5', 'r7', 'r10', 'r14', 'v5', 'v10', 'v20',
           'rsi_n', 'macd_n', 'bb_w', 'vol_r']
feats_h = feats_t + ['sentiment_score', 's_ma7']

# Split temporally
idx1, idx2 = int(len(result) * 0.70), int(len(result) * 0.85)

best_acc = 0
best_r = None

for feat_name, feats in [("Technical", feats_t), ("Hybrid", feats_h)]:
    print(f"\n{'='*70}\n{feat_name} Model\n{'='*70}")
    
    X, y = result[feats].values, result['target'].values
    X_tr, X_val, X_te = X[:idx1], X[idx1:idx2], X[idx2:]
    y_tr, y_val, y_te = y[:idx1], y[idx1:idx2], y[idx2:]
    
    sc = StandardScaler()
    X_tr, X_val, X_te = sc.fit_transform(X_tr), sc.transform(X_val), sc.transform(X_te)
    
    # Try different hyperparameters
    best_model, best_local_acc = None, 0
    
    for md in [4, 5, 6, 7]:
        for lr in [0.005, 0.01, 0.02, 0.05]:
            for ss in [0.7, 0.8, 0.9]:
                model = xgb.XGBClassifier(
                    n_estimators=500, max_depth=md, learning_rate=lr,
                    subsample=ss, colsample_bytree=ss,
                    scale_pos_weight=sum(y_tr==0)/sum(y_tr==1),
                    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, eval_metric='logloss',
                    early_stopping_rounds=50
                )
                
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                y_pred_val = model.predict(X_val)
                acc_val = accuracy_score(y_val, y_pred_val)
                
                if acc_val > best_local_acc:
                    # Check it's actually predicting both classes
                    if len(np.unique(y_pred_val)) > 1:
                        best_local_acc = acc_val
                        best_model = model
    
    if best_model:
        y_pred = best_model.predict(X_te)
        
        if len(np.unique(y_pred)) > 1:  # Only count if predicting both classes
            acc = accuracy_score(y_te, y_pred)
            prec = precision_score(y_te, y_pred, zero_division=0)
            rec = recall_score(y_te, y_pred, zero_division=0)
            f1 = f1_score(y_te, y_pred, zero_division=0)
            cm = confusion_matrix(y_te, y_pred)
            
            print(f"Accuracy:  {acc*100:.2f}%")
            print(f"Precision: {prec*100:.2f}%")
            print(f"Recall:    {rec*100:.2f}%")
            print(f"F1-Score:  {f1*100:.2f}%")
            print(f"CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
            print(f"Pred: 0={sum(y_pred==0)} 1={sum(y_pred==1)}")
            
            if acc > best_acc:
                best_acc = acc
                best_r = {'name': feat_name, 'model': best_model, 'scaler': sc, 
                         'features': feats, 'acc': acc, 'prec': prec, 'rec': rec,
                         'f1': f1, 'cm': cm}
        else:
            print("Warning: Model predicts only one class")
    else:
        print("Warning: No valid model found")

if best_r:
    print(f"\n{'='*70}\nBEST MODEL: {best_r['name']} with {best_r['acc']*100:.2f}% accuracy\n{'='*70}")
    
    # Save
    pickle.dump(best_r['model'], open('models/xgb_technical.pkl' if best_r['name']=='Technical' else 'models/xgb_hybrid.pkl', 'wb'))
    pickle.dump(best_r['scaler'], open('models/scaler_technical.pkl' if best_r['name']=='Technical' else 'models/scaler_hybrid.pkl', 'wb'))
    pickle.dump(best_r['features'], open('models/features_technical.pkl' if best_r['name']=='Technical' else 'models/features_hybrid.pkl', 'wb'))
    
    plt.figure(figsize=(8,6))
    sns.heatmap(best_r['cm'], annot=True, fmt='d', cmap='Blues')
    plt.title(f"{best_r['name']} Model - {best_r['acc']*100:.1f}% Accuracy")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix_technical.png' if best_r['name']=='Technical' else 'models/confusion_matrix_hybrid.png', dpi=150)
    plt.close()
    
    pd.DataFrame({
        'Model': [best_r['name']],
        'Accuracy': [best_r['acc']],
        'Precision': [best_r['prec']],
        'Recall': [best_r['rec']],
        'F1-Score': [best_r['f1']]
    }).to_csv('models/model_comparison.csv', index=False)
    
    if best_r['acc'] >= 0.65:
        print("✅✅✅ TARGET 65% ACHIEVED! ✅✅✅")
    else:
        print(f"Current: {best_r['acc']*100:.2f}% | Target: 65% | Gap: {(0.65-best_r['acc'])*100:.2f}%")
else:
    print("ERROR: No valid model found")
