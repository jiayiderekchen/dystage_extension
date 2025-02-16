import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_processor import DataProcessor
from model import TransformerEncoder
from trainer import ModelTrainer, StockDataset
from torch.utils.data import DataLoader
import os
import argparse
from datetime import datetime

def main():
    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['new', 'load', 'auto'], default='auto',
                       help='''Data processing mode: 
                       "new" - process data from scratch, 
                       "load" - use existing processed data,
                       "auto" - use processed data if exists, otherwise process from scratch''')
    parser.add_argument('--data_path', type=str, default='daily_clean_ret_v2_with_news.csv',
                       help='Path to raw data file')
    parser.add_argument('--processed_path', type=str, default='processed_data.pkl',
                       help='Path to processed data file')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'learning_rate': 1e-5,
        'batch_size': 16,
        'epochs': 10,
        'early_stopping': 10,
    }
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Process data
    processor = DataProcessor()
    
    if args.mode == 'new':
        # Force new processing
        print("Processing data from scratch...")
        if os.path.exists(args.processed_path):
            os.remove(args.processed_path)
        df = processor.process_data(args.data_path, save_processed=True, 
                                  processed_data_path=args.processed_path)
    
    elif args.mode == 'load':
        # Force loading existing processed data
        if not os.path.exists(args.processed_path):
            raise FileNotFoundError(f"No processed data found at {args.processed_path}")
        print(f"Loading processed data from {args.processed_path}")
        df = pd.read_pickle(args.processed_path)
    
    else:  # 'auto' mode
        # Use processed data if exists, otherwise process from scratch
        df = processor.process_data(args.data_path, save_processed=True, 
                                  processed_data_path=args.processed_path)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in 
                   ['datadate', 'cusip', 'ticker', 'ret', 'news_text']]
    X = df[feature_cols].values
    y = df['ret'].values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = TransformerEncoder(input_dim=input_dim)
    
    # Train model
    trainer = ModelTrainer(model, config)
    trainer.train(train_loader, val_loader)
    
    # After training and evaluation, compute feature importance using SHAP
    print("\nComputing SHAP feature importance...")
    importance_df, shap_values = trainer.compute_feature_importance(train_loader, feature_cols)
    
    # Save feature importance
    importance_path = 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to {importance_path}")
    
    # Print top 10 most important features
    print("\nTop 10 most important features:")
    print(importance_df.head(10))
    print("\nSHAP summary plot saved as 'shap_summary_plot.png'")
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    print("\nTest Set Metrics:")
    print(f"MSE: {test_metrics['mse']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"R2: {test_metrics['r2']:.4f}")
    
    # Save test metrics with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_df = pd.DataFrame({
        'timestamp': [timestamp],
        'mse': [test_metrics['mse']],
        'mae': [test_metrics['mae']],
        'r2': [test_metrics['r2']]
    })
    
    # Create or append to metrics file
    metrics_file = 'test_metrics_history.csv'
    if os.path.exists(metrics_file):
        metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)
    
    print(f"\nTest metrics saved to {metrics_file}")

if __name__ == "__main__":
    main() 