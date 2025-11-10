"""
Model Training Script
Trains XGBoost models with technical indicators and hybrid (technical + sentiment)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

def prepare_features(df, include_sentiment=False):
    """Prepare feature matrix and target variable"""
    
    # Technical indicator features
    technical_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
        'EMA_20', 'SMA_50',
        'BB_high', 'BB_low', 'BB_mid',
        'Price_Change', 'Volume_Change'
    ]
    
    if include_sentiment:
        features = technical_features + ['sentiment_score']
        print(f"Using {len(features)} features (technical + sentiment)")
    else:
        features = technical_features
        print(f"Using {len(features)} features (technical only)")
    
    X = df[features].copy()
    y = df['target'].copy()
    
    return X, y, features

def train_model(X_train, y_train, X_val, y_val, model_name):
    """Train XGBoost model"""
    print(f"\nTraining {model_name}...")
    
    # Create XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"✓ {model_name} trained successfully")
    
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
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {model_name}')
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
        'Model': ['Technical Only', 'Hybrid (Technical + Sentiment)'],
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
    print("Model Training - Indian Stock Market Prediction")
    print("=" * 60)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Split data: 70% train, 15% validation, 15% test
    print("\nSplitting data...")
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=False)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=False)
    
    print(f"  Training set:   {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation set: {len(val_df)} records ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test set:       {len(test_df)} records ({len(test_df)/len(df)*100:.1f}%)")
    
    # Model A: Technical indicators only
    print("\n" + "=" * 60)
    print("MODEL A: TECHNICAL INDICATORS ONLY")
    print("=" * 60)
    
    X_train_tech, y_train, features_tech = prepare_features(train_df, include_sentiment=False)
    X_val_tech, y_val, _ = prepare_features(val_df, include_sentiment=False)
    X_test_tech, y_test, _ = prepare_features(test_df, include_sentiment=False)
    
    model_tech = train_model(X_train_tech, y_train, X_val_tech, y_val, "XGBoost Technical")
    results_tech = evaluate_model(model_tech, X_test_tech, y_test, "XGBoost Technical")
    
    # Save model
    with open('models/xgb_technical.pkl', 'wb') as f:
        pickle.dump(model_tech, f)
    print("✓ Model saved to models/xgb_technical.pkl")
    
    # Plot confusion matrix
    plot_confusion_matrix(results_tech['confusion_matrix'], "XGBoost Technical", 
                         'models/confusion_matrix_technical.png')
    
    # Plot feature importance
    plot_feature_importance(model_tech, features_tech, "XGBoost Technical",
                           'models/feature_importance_technical.png')
    
    # Model B: Hybrid (Technical + Sentiment)
    print("\n" + "=" * 60)
    print("MODEL B: HYBRID (TECHNICAL + SENTIMENT)")
    print("=" * 60)
    
    X_train_hybrid, y_train, features_hybrid = prepare_features(train_df, include_sentiment=True)
    X_val_hybrid, y_val, _ = prepare_features(val_df, include_sentiment=True)
    X_test_hybrid, y_test, _ = prepare_features(test_df, include_sentiment=True)
    
    model_hybrid = train_model(X_train_hybrid, y_train, X_val_hybrid, y_val, "XGBoost Hybrid")
    results_hybrid = evaluate_model(model_hybrid, X_test_hybrid, y_test, "XGBoost Hybrid")
    
    # Save model
    with open('models/xgb_hybrid.pkl', 'wb') as f:
        pickle.dump(model_hybrid, f)
    print("✓ Model saved to models/xgb_hybrid.pkl")
    
    # Plot confusion matrix
    plot_confusion_matrix(results_hybrid['confusion_matrix'], "XGBoost Hybrid",
                         'models/confusion_matrix_hybrid.png')
    
    # Plot feature importance
    plot_feature_importance(model_hybrid, features_hybrid, "XGBoost Hybrid",
                           'models/feature_importance_hybrid.png')
    
    # Compare models
    comparison = compare_models(results_tech, results_hybrid)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - models/xgb_technical.pkl")
    print("  - models/xgb_hybrid.pkl")
    print("  - models/confusion_matrix_technical.png")
    print("  - models/confusion_matrix_hybrid.png")
    print("  - models/feature_importance_technical.png")
    print("  - models/feature_importance_hybrid.png")
    print("  - models/model_comparison.csv")

if __name__ == "__main__":
    main()
