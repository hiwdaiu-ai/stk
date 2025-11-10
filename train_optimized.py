"""
Advanced Model Training with Multiple Strategies to Achieve 65%+ Accuracy
Uses LSTM, better feature engineering, and class weighting
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_engineer_advanced_features():
    """Load data and create advanced features for better prediction"""
    print("Loading and engineering advanced features...")
    
    df = pd.read_csv('data/processed/features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    result_dfs = []
    
    for symbol in df['Symbol'].unique():
        symbol_df = df[df['Symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('Date')
        
        # Price momentum - multiple timeframes
        for period in [2, 3, 5, 7, 10, 14, 20, 30]:
            symbol_df[f'returns_{period}d'] = symbol_df['Close'].pct_change(period)
            symbol_df[f'high_low_diff_{period}d'] = (symbol_df['High'] - symbol_df['Low']).rolling(period).mean()
        
        # Volatility measures
        for period in [5, 10, 20, 30]:
            symbol_df[f'volatility_{period}d'] = symbol_df['Close'].pct_change().rolling(period).std()
            symbol_df[f'volume_volatility_{period}d'] = symbol_df['Volume'].pct_change().rolling(period).std()
        
        # Moving average crossovers and divergences
        symbol_df['ema_sma_diff'] = symbol_df['EMA_20'] - symbol_df['SMA_50']
        symbol_df['price_ema_diff'] = symbol_df['Close'] - symbol_df['EMA_20']
        symbol_df['price_sma_diff'] = symbol_df['Close'] - symbol_df['SMA_50']
        
        # Price position relative to MAs
        symbol_df['price_vs_ema20'] = (symbol_df['Close'] - symbol_df['EMA_20']) / (symbol_df['EMA_20'] + 1e-10)
        symbol_df['price_vs_sma50'] = (symbol_df['Close'] - symbol_df['SMA_50']) / (symbol_df['SMA_50'] + 1e-10)
        
        # RSI advanced features
        symbol_df['rsi_normalized'] = symbol_df['RSI'] / 100
        symbol_df['rsi_overbought'] = (symbol_df['RSI'] > 70).astype(int)
        symbol_df['rsi_oversold'] = (symbol_df['RSI'] < 30).astype(int)
        symbol_df['rsi_momentum'] = symbol_df['RSI'].diff(5)
        symbol_df['rsi_acceleration'] = symbol_df['RSI'].diff(5).diff(5)
        
        # MACD advanced
        symbol_df['macd_histogram'] = symbol_df['MACD'] - symbol_df['MACD_signal']
        symbol_df['macd_signal_cross'] = ((symbol_df['MACD'] > symbol_df['MACD_signal']) & 
                                          (symbol_df['MACD'].shift(1) <= symbol_df['MACD_signal'].shift(1))).astype(int)
        symbol_df['macd_strength'] = symbol_df['MACD_diff'] / (symbol_df['Close'] + 1e-10)
        symbol_df['macd_momentum'] = symbol_df['MACD'].diff(5)
        
        # Bollinger Bands advanced
        bb_width = symbol_df['BB_high'] - symbol_df['BB_low']
        symbol_df['bb_width'] = bb_width / (symbol_df['Close'] + 1e-10)
        symbol_df['bb_position'] = (symbol_df['Close'] - symbol_df['BB_low']) / (bb_width + 1e-10)
        symbol_df['bb_squeeze'] = (bb_width < bb_width.rolling(20).mean()).astype(int)
        symbol_df['price_above_bb_high'] = (symbol_df['Close'] > symbol_df['BB_high']).astype(int)
        symbol_df['price_below_bb_low'] = (symbol_df['Close'] < symbol_df['BB_low']).astype(int)
        
        # Volume advanced
        symbol_df['volume_ma5'] = symbol_df['Volume'].rolling(5).mean()
        symbol_df['volume_ma20'] = symbol_df['Volume'].rolling(20).mean()
        symbol_df['volume_ratio_5'] = symbol_df['Volume'] / (symbol_df['volume_ma5'] + 1)
        symbol_df['volume_ratio_20'] = symbol_df['Volume'] / (symbol_df['volume_ma20'] + 1)
        symbol_df['volume_spike'] = (symbol_df['Volume'] > 2 * symbol_df['volume_ma20']).astype(int)
        
        # Trend indicators
        for period in [5, 10, 20]:
            symbol_df[f'close_ma{period}'] = symbol_df['Close'].rolling(period).mean()
        symbol_df['trend_5_10'] = (symbol_df['close_ma5'] > symbol_df['close_ma10']).astype(int)
        symbol_df['trend_10_20'] = (symbol_df['close_ma10'] > symbol_df['close_ma20']).astype(int)
        symbol_df['trend_strength'] = (symbol_df['close_ma5'] - symbol_df['close_ma20']) / (symbol_df['close_ma20'] + 1e-10)
        
        # Price range features
        symbol_df['hl_ratio'] = (symbol_df['High'] - symbol_df['Low']) / (symbol_df['Close'] + 1e-10)
        symbol_df['oc_ratio'] = (symbol_df['Open'] - symbol_df['Close']) / (symbol_df['Close'] + 1e-10)
        symbol_df['body_length'] = abs(symbol_df['Open'] - symbol_df['Close']) / (symbol_df['Close'] + 1e-10)
        
        # Lag features - previous day patterns
        for lag in [1, 2, 3, 5]:
            symbol_df[f'close_lag{lag}'] = symbol_df['Close'].shift(lag)
            symbol_df[f'volume_lag{lag}'] = symbol_df['Volume'].shift(lag)
            symbol_df[f'rsi_lag{lag}'] = symbol_df['RSI'].shift(lag)
        
        # Day of week effect
        symbol_df['day_of_week'] = symbol_df['Date'].dt.dayofweek
        symbol_df['is_monday'] = (symbol_df['day_of_week'] == 0).astype(int)
        symbol_df['is_friday'] = (symbol_df['day_of_week'] == 4).astype(int)
        
        # Sentiment features - enhanced
        symbol_df['sentiment_ma3'] = symbol_df['sentiment_score'].rolling(3).mean()
        symbol_df['sentiment_ma7'] = symbol_df['sentiment_score'].rolling(7).mean()
        symbol_df['sentiment_ma14'] = symbol_df['sentiment_score'].rolling(14).mean()
        symbol_df['sentiment_positive'] = (symbol_df['sentiment_score'] > 0.05).astype(int)
        symbol_df['sentiment_negative'] = (symbol_df['sentiment_score'] < -0.05).astype(int)
        symbol_df['sentiment_momentum'] = symbol_df['sentiment_score'].diff(3)
        symbol_df['sentiment_volatility'] = symbol_df['sentiment_score'].rolling(7).std()
        
        # Combined indicators
        symbol_df['momentum_score'] = (
            symbol_df['returns_5d'] * 0.3 + 
            symbol_df['returns_10d'] * 0.3 + 
            symbol_df['rsi_normalized'] * 0.2 +
            symbol_df['macd_strength'] * 0.2
        )
        
        result_dfs.append(symbol_df)
    
    combined_df = pd.concat(result_dfs, ignore_index=True)
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
    combined_df = combined_df.dropna()
    
    print(f"✓ Advanced features engineered: {len(combined_df)} records")
    print(f"✓ Total features: {len(combined_df.columns)}")
    
    return combined_df

def prepare_datasets_advanced(df, include_sentiment=False):
    """Prepare comprehensive feature sets"""
    
    # Core technical features
    core_features = [
        'RSI', 'MACD', 'MACD_diff', 'EMA_20', 'SMA_50',
        'Price_Change', 'Volume_Change',
        'BB_high', 'BB_low', 'BB_mid'
    ]
    
    # Momentum features
    momentum_features = [
        'returns_2d', 'returns_3d', 'returns_5d', 'returns_7d', 'returns_10d', 'returns_14d', 'returns_20d',
        'volatility_5d', 'volatility_10d', 'volatility_20d'
    ]
    
    # Price vs MA features
    price_ma_features = [
        'price_vs_ema20', 'price_vs_sma50', 
        'ema_sma_diff', 'price_ema_diff', 'price_sma_diff'
    ]
    
    # RSI advanced
    rsi_features = [
        'rsi_normalized', 'rsi_overbought', 'rsi_oversold', 
        'rsi_momentum', 'rsi_acceleration'
    ]
    
    # MACD advanced
    macd_features = [
        'macd_histogram', 'macd_signal_cross', 'macd_strength', 'macd_momentum'
    ]
    
    # Bollinger advanced
    bb_features = [
        'bb_width', 'bb_position', 'bb_squeeze', 
        'price_above_bb_high', 'price_below_bb_low'
    ]
    
    # Volume features
    volume_features = [
        'volume_ratio_5', 'volume_ratio_20', 'volume_spike',
        'volume_volatility_10d'
    ]
    
    # Trend features
    trend_features = [
        'trend_5_10', 'trend_10_20', 'trend_strength'
    ]
    
    # Price features
    price_features = [
        'hl_ratio', 'oc_ratio', 'body_length'
    ]
    
    # Lag features
    lag_features = [
        'close_lag1', 'close_lag2', 'rsi_lag1', 'rsi_lag2'
    ]
    
    # Time features
    time_features = [
        'day_of_week', 'is_monday', 'is_friday'
    ]
    
    # Momentum score
    combined_features = ['momentum_score']
    
    # Build feature list
    features = (core_features + momentum_features + price_ma_features + 
                rsi_features + macd_features + bb_features + volume_features +
                trend_features + price_features + lag_features + time_features +
                combined_features)
    
    if include_sentiment:
        sentiment_features = [
            'sentiment_score', 'sentiment_ma3', 'sentiment_ma7', 'sentiment_ma14',
            'sentiment_positive', 'sentiment_negative', 
            'sentiment_momentum', 'sentiment_volatility'
        ]
        features = features + sentiment_features
    
    print(f"Using {len(features)} features")
    
    X = df[features].values
    y = df['target'].values
    
    # Temporal split: 70-15-15
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, shuffle=False)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, features

def train_advanced_ensemble(X_train, y_train, X_val, y_val, model_name):
    """Train advanced ensemble with multiple algorithms"""
    print(f"\nTraining {model_name}...")
    
    # Calculate class weights
    n_samples = len(y_train)
    n_class_0 = np.sum(y_train == 0)
    n_class_1 = np.sum(y_train == 1)
    weight_0 = n_samples / (2 * n_class_0)
    weight_1 = n_samples / (2 * n_class_1)
    sample_weights = np.where(y_train == 0, weight_0, weight_1)
    
    # XGBoost with optimized parameters
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0.01,
        reg_lambda=0.5,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=weight_1/weight_0
    )
    
    # RandomForest with class weight
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.7,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # AdaBoost
    ada_model = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.5,
        random_state=42
    )
    
    # SVM with probability
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    
    # Voting Ensemble with more models
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('rf', rf_model),
            ('gb', gb_model),
            ('ada', ada_model),
            ('svm', svm_model)
        ],
        voting='soft',
        weights=[3, 2, 2, 1, 1],  # Give more weight to XGB and RF
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
    
    print(f"\n{'='*70}")
    print(f"{model_name} Performance:")
    print(f"{'='*70}")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"{'='*70}")
    
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
    """Main training pipeline"""
    print("=" * 70)
    print("ADVANCED MODEL TRAINING - TARGET: 65%+ ACCURACY")
    print("=" * 70)
    
    os.makedirs('models', exist_ok=True)
    
    # Load and engineer features
    df = load_and_engineer_advanced_features()
    
    print(f"\nDataset: {len(df)} records")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    best_accuracy = 0
    best_model_name = None
    
    # Model A: Technical features
    print("\n" + "=" * 70)
    print("MODEL A: ADVANCED ENSEMBLE WITH TECHNICAL FEATURES")
    print("=" * 70)
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_tech, features_tech = prepare_datasets_advanced(df, include_sentiment=False)
    
    print(f"Training set: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    model_tech = train_advanced_ensemble(X_train, y_train, X_val, y_val, "Technical Advanced Ensemble")
    results_tech = evaluate_model(model_tech, X_test, y_test, "Technical Advanced Ensemble")
    
    if results_tech['accuracy'] > best_accuracy:
        best_accuracy = results_tech['accuracy']
        best_model_name = "Technical Advanced Ensemble"
    
    with open('models/xgb_technical.pkl', 'wb') as f:
        pickle.dump(model_tech, f)
    with open('models/scaler_technical.pkl', 'wb') as f:
        pickle.dump(scaler_tech, f)
    with open('models/features_technical.pkl', 'wb') as f:
        pickle.dump(features_tech, f)
    
    plot_confusion_matrix(results_tech['confusion_matrix'], "Technical Advanced Ensemble", 
                         'models/confusion_matrix_technical.png')
    
    # Model B: Hybrid with sentiment
    print("\n" + "=" * 70)
    print("MODEL B: ADVANCED ENSEMBLE WITH TECHNICAL + SENTIMENT")
    print("=" * 70)
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_hybrid, features_hybrid = prepare_datasets_advanced(df, include_sentiment=True)
    
    print(f"Training set: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    model_hybrid = train_advanced_ensemble(X_train, y_train, X_val, y_val, "Hybrid Advanced Ensemble")
    results_hybrid = evaluate_model(model_hybrid, X_test, y_test, "Hybrid Advanced Ensemble")
    
    if results_hybrid['accuracy'] > best_accuracy:
        best_accuracy = results_hybrid['accuracy']
        best_model_name = "Hybrid Advanced Ensemble"
    
    with open('models/xgb_hybrid.pkl', 'wb') as f:
        pickle.dump(model_hybrid, f)
    with open('models/scaler_hybrid.pkl', 'wb') as f:
        pickle.dump(scaler_hybrid, f)
    with open('models/features_hybrid.pkl', 'wb') as f:
        pickle.dump(features_hybrid, f)
    
    plot_confusion_matrix(results_hybrid['confusion_matrix'], "Hybrid Advanced Ensemble",
                         'models/confusion_matrix_hybrid.png')
    
    # Comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    comparison = pd.DataFrame({
        'Model': ['Technical Advanced Ensemble', 'Hybrid Advanced Ensemble'],
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
    print(f"✅ BEST MODEL: {best_model_name}")
    print(f"✅ BEST ACCURACY: {best_accuracy*100:.2f}%")
    
    if best_accuracy >= 0.65:
        print("🎉 TARGET ACHIEVED: 65%+ ACCURACY!")
    else:
        print(f"⚠️  Current: {best_accuracy*100:.2f}% | Target: 65% | Gap: {(0.65-best_accuracy)*100:.2f}%")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
