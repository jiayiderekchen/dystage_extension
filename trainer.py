import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from tqdm import tqdm
import pandas as pd
import shap
import matplotlib.pyplot as plt

class StockDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class ModelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate']
        )
        
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for features, targets in tqdm(train_loader, desc="Training"):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        return {
            'loss': total_loss / len(val_loader),
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
    
    def save_checkpoint(self, epoch, metrics, path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def train(self, train_loader, val_loader):
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val MSE: {val_metrics['mse']:.4f}")
            print(f"Val MAE: {val_metrics['mae']:.4f}")
            print(f"Val R2: {val_metrics['r2']:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(
                epoch,
                val_metrics,
                f"checkpoints/model_epoch_{epoch+1}.pt"
            )
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config['early_stopping']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break 
    
    def compute_feature_importance(self, train_loader, feature_names, n_background=100):
        """Compute feature importance using SHAP"""
        self.model.eval()
        
        # Convert train_loader to numpy array for SHAP
        background_data = []
        for features, _ in train_loader:
            background_data.append(features.numpy())
            if len(background_data) * features.shape[0] >= n_background:
                break
        background_data = np.concatenate(background_data)[:n_background]
        
        # Define model wrapper for SHAP
        def model_wrapper(x):
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(self.device)
                return self.model(x_tensor).cpu().numpy()
        
        # Initialize SHAP explainer
        explainer = shap.DeepExplainer(model_wrapper, background_data)
        
        # Calculate SHAP values
        shap_values = []
        for features, _ in tqdm(train_loader, desc="Computing SHAP values"):
            batch_shap_values = explainer.shap_values(features.numpy())
            shap_values.append(batch_shap_values)
        
        # Combine all SHAP values
        shap_values = np.concatenate(shap_values)
        
        # Calculate feature importance based on mean absolute SHAP values
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Create DataFrame with feature names and importance scores
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Create SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, background_data, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png')
        plt.close()
        
        return importance_df, shap_values 