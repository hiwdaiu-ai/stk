"""
Main Pipeline Script
Runs the complete stock prediction workflow
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and print status"""
    print("\n" + "=" * 70)
    print(f"🚀 {description}")
    print("=" * 70)
    
    result = subprocess.run(command, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed")
        return False
    
    print(f"\n✅ {description} completed successfully")
    return True

def main():
    """Run complete pipeline"""
    print("\n" + "=" * 70)
    print("INDIAN STOCK MARKET PREDICTION - COMPLETE PIPELINE")
    print("=" * 70)
    
    steps = [
        ("python collect_data.py", "Step 1: Data Collection"),
        ("python feature_engineering.py", "Step 2: Feature Engineering"),
        ("python train_models.py", "Step 3: Model Training"),
    ]
    
    for command, description in steps:
        if not run_command(command, description):
            print("\n❌ Pipeline failed. Please check the errors above.")
            sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  📁 data/raw/")
    print("     - technical_data.csv")
    print("     - news_data.csv")
    print("  📁 data/processed/")
    print("     - features.csv")
    print("  📁 models/")
    print("     - xgb_technical.pkl")
    print("     - xgb_hybrid.pkl")
    print("     - confusion_matrix_technical.png")
    print("     - confusion_matrix_hybrid.png")
    print("     - feature_importance_technical.png")
    print("     - feature_importance_hybrid.png")
    print("     - model_comparison.csv")
    print("\n" + "=" * 70)
    print("📊 To view the dashboard, run:")
    print("   streamlit run app/dashboard.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
