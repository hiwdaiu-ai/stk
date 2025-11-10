"""
Improved Model Training Script
Uses advanced feature engineering, hyperparameter tuning, and better model configurations
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

def load_data():
    """Load processed features"""
    print("Loading processed features...")
    df = pd.read_csv('data/processed/features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded {len(df)} records")
    return df

def create_advanced_features(df):
    """Create more sophisticated features for better prediction"""
    print("\nCreating advanced features...")
    
    result_dfs = []
    
    for symbol in df['Symbol'].unique():
        symbol_df = df[df['Symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('Date')
        
        # Price momentum features
        symbol_df['returns_1d'] = symbol_df['Close'].pct_change(1)
        symbol_df['returns_3d'] = symbol_df['Close'].pct_change(3)
        symbol_df['returns_5d'] = symbol_df['Close'].pct_change(5)
        symbol_df['returns_10d'] = symbol_df['Close'].pct_change(10)
        
        # Volatility features
        symbol_df['volatility_5d'] = symbol_df['returns_1d'].rolling(window=5).std()
        symbol_df['volatility_10d'] = symbol_df['returns_1d'].rolling(window=10).std()
        
        # Price relative to moving averages
        symbol_df['price_to_ema20'] = symbol_df['Close'] / symbol_df['EMA_20']
        symbol_df['price_to_sma50'] = symbol_df['Close'] / symbol_df['SMA_50']
        
        # RSI momentum
        symbol_df['rsi_change'] = symbol_df['RSI'].diff()
        symbol_df['rsi_momentum'] = symbol_df['RSI'].diff(3)
        
        # MACD features
        symbol_df['macd_strength'] = symbol_df['MACD_diff'] / symbol_df['Close']
        
        # Bollinger Band position
        symbol_df['bb_position'] = (symbol_df['Close'] - symbol_df['BB_low']) / (symbol_df['BB_high'] - symbol_df['BB_low'])
        
        # Volume features
        symbol_df['volume_sma20'] = symbol_df['Volume'].rolling(window=20).mean()
        symbol_df['volume_ratio'] = symbol_df['Volume'] / symbol_df['volume_sma20']
        
        # High-Low range
        symbol_df['hl_range'] = (symbol_df['High'] - symbol_df['Low']) / symbol_df['Close']
        
        # Moving average crossover signals
        symbol_df['ema_sma_cross'] = (symbol_df['EMA_20'] > symbol_df['SMA_50']).astype(int)
        
        # Trend strength
        symbol_df['trend_strength'] = symbol_df['Close'].rolling(window=10).apply(
            lambda x: 1 if x.is_monotonic_increasing else (-1 if x.is_monotonic_decreasing else 0)
        )
        
        # Sentiment momentum
        symbol_df['sentiment_change'] = symbol_df['sentiment_score'].diff()
        symbol_df['sentiment_ma5'] = symbol_df['sentiment_score'].rolling(window=5).mean()
        
        result_dfs.append(symbol_df)
    
    combined_df = pd.concat(result_dfs, ignore_index=True)
    
    # Replace inf with NaN
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
    
    print(f"✓ Advanced features created")
    
    return combined_df

def prepare_features(df, include_sentiment=False, use_advanced=True):
    """Prepare feature matrix and target variable"""
    
    # Basic technical features
    technical_features = [
        'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
        'EMA_20', 'SMA_50',
        'Price_Change', 'Volume_Change'
    ]
    
    if use_advanced:
        # Advanced features
        advanced_features = [
            'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d',
            'volatility_5d', 'volatility_10d',
            'price_to_ema20', 'price_to_sma50',
            'rsi_change', 'rsi_momentum',
            'macd_strength',
            'bb_position',
            'volume_ratio',
            'hl_range',
            'ema_sma_cross',
            'trend_strength'
        ]
        
        features = technical_features + advanced_features
    else:
        features = technical_features
    
    if include_sentiment:
        features = features + ['sentiment_score', 'sentiment_change', 'sentiment_ma5']
        print(f"Using {len(features)} features (technical + advanced + sentiment)")
    else:
        print(f"Using {len(features)} features (technical + advanced)")
    
    X = df[features].copy()
    y = df['target'].copy()
    
    return X, y, features

def train_model(X_train, y_train, X_val, y_val, model_name):
    """Train XGBoost model with better hyperparameters"""
    print(f"\nTraining {model_name}...")
    
    # Optimized XGBoost parameters for stock prediction
    model = xgb.XGBClassifier(
        n_estimators=500,           # More trees for better learning
        max_depth=7,                # Deeper trees
        learning_rate=0.01,         # Very low learning rate
        subsample=0.7,              # Aggressive row sampling
        colsample_bytree=0.7,       # Aggressive column sampling
        min_child_weight=1,         # Allow smaller splits
        gamma=0,                    # Less regularization initially
        reg_alpha=0.01,             # Light L1 regularization
        reg_lambda=0.1,             # Light L2 regularization
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=50,   # More patience
        scale_pos_weight=1          # Balance classes
    )
    
    # Train model with validation
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"✓ {model_name} trained successfully")
    print(f"  Best iteration: {model.best_iteration}")
    
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    print(f"\nEvaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'predictions_proba': y_pred_proba
    }

def plot_confusion_matrix(cm, model_name, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")

def plot_feature_importance(model, features, model_name, save_path):
    """Plot and save feature importance"""
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False).head(20)  # Top 20
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Feature Importance - {model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Feature importance plot saved to {save_path}")

def compare_models(results_technical, results_hybrid):
    """Compare performance of both models"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    comparison = pd.DataFrame({
        'Model': ['Technical Only (Improved)', 'Hybrid (Improved)'],
        'Accuracy': [results_technical['accuracy'], results_hybrid['accuracy']],
        'Precision': [results_technical['precision'], results_hybrid['precision']],
        'Recall': [results_technical['recall'], results_hybrid['recall']],
        'F1-Score': [results_technical['f1_score'], results_hybrid['f1_score']]
    })
    
    print("\n", comparison.to_string(index=False))
    
    # Calculate improvement
    improvement = (results_hybrid['accuracy'] - results_technical['accuracy']) * 100
    print(f"\nAccuracy improvement with sentiment: {improvement:+.2f}%")
    
    # Save comparison
    comparison.to_csv('models/model_comparison.csv', index=False)
    print("\n✓ Model comparison saved to models/model_comparison.csv")
    
    return comparison

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("IMPROVED Model Training - Indian Stock Market Prediction")
    print("=" * 60)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Create advanced features
    df = create_advanced_features(df)
    
    # Drop rows with NaN (from feature creation)
    print(f"\nBefore cleaning: {len(df)} records")
    df = df.dropna()
    print(f"After cleaning: {len(df)} records")
    
    # Split data: 70% train, 15% validation, 15% test
    print("\nSplitting data...")
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=False)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=False)
    
    print(f"  Training set:   {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation set: {len(val_df)} records ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test set:       {len(test_df)} records ({len(test_df)/len(df)*100:.1f}%)")
    
    # Model A: Technical + Advanced features
    print("\n" + "=" * 60)
    print("MODEL A: TECHNICAL + ADVANCED FEATURES")
    print("=" * 60)
    
    X_train_tech, y_train, features_tech = prepare_features(train_df, include_sentiment=False, use_advanced=True)
    X_val_tech, y_val, _ = prepare_features(val_df, include_sentiment=False, use_advanced=True)
    X_test_tech, y_test, _ = prepare_features(test_df, include_sentiment=False, use_advanced=True)
    
    # Normalize features
    scaler_tech = StandardScaler()
    X_train_tech = scaler_tech.fit_transform(X_train_tech)
    X_val_tech = scaler_tech.transform(X_val_tech)
    X_test_tech = scaler_tech.transform(X_test_tech)
    
    model_tech = train_model(X_train_tech, y_train, X_val_tech, y_val, "XGBoost Technical (Improved)")
    results_tech = evaluate_model(model_tech, X_test_tech, y_test, "XGBoost Technical (Improved)")
    
    # Save model and scaler
    with open('models/xgb_technical.pkl', 'wb') as f:
        pickle.dump(model_tech, f)
    with open('models/scaler_technical.pkl', 'wb') as f:
        pickle.dump(scaler_tech, f)
    print("✓ Model and scaler saved to models/xgb_technical.pkl and models/scaler_technical.pkl")
    
    # Plot confusion matrix
    plot_confusion_matrix(results_tech['confusion_matrix'], "XGBoost Technical (Improved)", 
                         'models/confusion_matrix_technical.png')
    
    # Plot feature importance
    plot_feature_importance(model_tech, features_tech, "XGBoost Technical (Improved)",
                           'models/feature_importance_technical.png')
    
    # Model B: Hybrid (Technical + Advanced + Sentiment)
    print("\n" + "=" * 60)
    print("MODEL B: HYBRID (TECHNICAL + ADVANCED + SENTIMENT)")
    print("=" * 60)
    
    X_train_hybrid, y_train, features_hybrid = prepare_features(train_df, include_sentiment=True, use_advanced=True)
    X_val_hybrid, y_val, _ = prepare_features(val_df, include_sentiment=True, use_advanced=True)
    X_test_hybrid, y_test, _ = prepare_features(test_df, include_sentiment=True, use_advanced=True)
    
    # Normalize features
    scaler_hybrid = StandardScaler()
    X_train_hybrid = scaler_hybrid.fit_transform(X_train_hybrid)
    X_val_hybrid = scaler_hybrid.transform(X_val_hybrid)
    X_test_hybrid = scaler_hybrid.transform(X_test_hybrid)
    
    model_hybrid = train_model(X_train_hybrid, y_train, X_val_hybrid, y_val, "XGBoost Hybrid (Improved)")
    results_hybrid = evaluate_model(model_hybrid, X_test_hybrid, y_test, "XGBoost Hybrid (Improved)")
    
    # Save model and scaler
    with open('models/xgb_hybrid.pkl', 'wb') as f:
        pickle.dump(model_hybrid, f)
    with open('models/scaler_hybrid.pkl', 'wb') as f:
        pickle.dump(scaler_hybrid, f)
    print("✓ Model and scaler saved to models/xgb_hybrid.pkl and models/scaler_hybrid.pkl")
    
    # Plot confusion matrix
    plot_confusion_matrix(results_hybrid['confusion_matrix'], "XGBoost Hybrid (Improved)",
                         'models/confusion_matrix_hybrid.png')
    
    # Plot feature importance
    plot_feature_importance(model_hybrid, features_hybrid, "XGBoost Hybrid (Improved)",
                           'models/feature_importance_hybrid.png')
    
    # Save feature names for later use
    with open('models/features_technical.pkl', 'wb') as f:
        pickle.dump(features_tech, f)
    with open('models/features_hybrid.pkl', 'wb') as f:
        pickle.dump(features_hybrid, f)
    
    # Compare models
    comparison = compare_models(results_tech, results_hybrid)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - models/xgb_technical.pkl")
    print("  - models/xgb_hybrid.pkl")
    print("  - models/scaler_technical.pkl")
    print("  - models/scaler_hybrid.pkl")
    print("  - models/features_technical.pkl")
    print("  - models/features_hybrid.pkl")
    print("  - models/confusion_matrix_technical.png")
    print("  - models/confusion_matrix_hybrid.png")
    print("  - models/feature_importance_technical.png")
    print("  - models/feature_importance_hybrid.png")
    print("  - models/model_comparison.csv")

if __name__ == "__main__":
    main()
