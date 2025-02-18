import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error  # for MAPE
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
        
        # Process batches in order
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
        
        # Convert to numpy arrays for metric calculation
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate RMSE and MAE
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        
        # Calculate MAPE with clipping
        def mape_clip(label, pred, threshold=0.1):
            v = np.clip(np.abs(label), threshold, None)
            diff = np.abs((label - pred) / v)
            return 100.0 * np.mean(diff)
        
        mape = mape_clip(all_targets, all_preds)
        
        return {
            'loss': total_loss / len(val_loader),
            'rmse': rmse,
            'mae': mae,
            'mape': mape
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
            print(f"Val RMSE: {val_metrics['rmse']:.4f}")
            print(f"Val MAE: {val_metrics['mae']:.4f}")
            print(f"Val MAPE: {val_metrics['mape']:.2f}%")
            
            # Save checkpoint
            self.save_checkpoint(
                epoch,
                val_metrics,
                f"checkpoints/model_epoch_{epoch+1}.pt"
            )
            
            # Save best model based on RMSE
            if val_metrics['rmse'] < self.best_val_loss:
                self.best_val_loss = val_metrics['rmse']
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config['early_stopping']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break 
    
    def compute_feature_importance(self, train_loader, feature_names):
        """
        Compute feature importance using SHAP GradientExplainer
        """
        # Set model to eval mode
        self.model.eval()
        
        # Create a wrapper for the model to reshape output
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                return self.model(x).unsqueeze(-1)  # Add extra dimension for SHAP
        
        wrapped_model = ModelWrapper(self.model)
        wrapped_model.to(self.device)
        
        # Get a batch of background data
        background_data = next(iter(train_loader))[0].to(self.device)
        
        # Initialize the SHAP explainer
        explainer = shap.GradientExplainer(
            model=wrapped_model,
            data=background_data,
        )
        
        # Get a smaller subset of data for SHAP analysis
        n_samples = min(100, len(train_loader.dataset))
        sample_indices = torch.randperm(len(train_loader.dataset))[:n_samples]
        sample_data = torch.stack([train_loader.dataset[i][0] for i in sample_indices]).to(self.device)
        
        try:
            # Calculate SHAP values
            shap_values = explainer.shap_values(sample_data)
            
            # If shap_values is a list (multiple outputs), take the first element
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Remove extra dimension if present
            if len(shap_values.shape) > 2:
                shap_values = shap_values.squeeze(-1)
            
            # Calculate mean absolute SHAP values for each feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Create DataFrame with feature importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_shap
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Create and save SHAP summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values, 
                sample_data.cpu().numpy(), 
                feature_names=feature_names, 
                show=False
            )
            plt.savefig('shap_summary_plot.png')
            plt.close()
            
            return importance_df, shap_values
            
        except Exception as e:
            print(f"Error in SHAP computation: {str(e)}")
            print("Falling back to simple feature importance calculation...")
            
            # Fallback to a simpler feature importance calculation
            importance = []
            with torch.no_grad():
                for i in range(sample_data.shape[1]):
                    # Zero out one feature at a time and measure the change in output
                    modified_data = sample_data.clone()
                    modified_data[:, i] = 0
                    original_output = self.model(sample_data)
                    modified_output = self.model(modified_data)
                    importance.append(torch.abs(original_output - modified_output).mean().item())
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df, None 