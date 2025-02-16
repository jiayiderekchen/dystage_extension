import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch
import os
import pickle

class DataProcessor:
    def __init__(self):
        self.finbert = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.finbert.to(self.device)
        
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def get_sentiment(self, texts):
        """Get FinBERT sentiment scores for a list of texts"""
        sentiments = []
        
        for text in tqdm(texts, desc="Processing sentiments"):
            if not text:
                sentiments.append({'pos': 0.0, 'neg': 0.0, 'neu': 1.0})
                continue
                
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.finbert(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                sentiment_dict = {
                    'pos': scores[0][0].item(),
                    'neg': scores[0][1].item(),
                    'neu': scores[0][2].item()
                }
                sentiments.append(sentiment_dict)
                
        return pd.DataFrame(sentiments)
    
    def normalize_features(self, df, feature_columns):
        """Normalize features to range (-1, 1)"""
        for col in feature_columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = 2 * (df[col] - min_val) / (max_val - min_val) - 1
        return df
    
    def process_data(self, data_path, save_processed=True, processed_data_path='processed_data.pkl'):
        """Main data processing pipeline with option to save processed data"""
        # Check if processed data exists
        if os.path.exists(processed_data_path):
            print(f"Loading preprocessed data from {processed_data_path}")
            return pd.read_pickle(processed_data_path)
            
        # Original processing logic
        print("Processing new data...")
        df = pd.read_csv(data_path, low_memory=False)
        
        # Process text data
        print("Processing news text...")
        df['news_text'] = df['news_text'].apply(self.preprocess_text)
        
        # Get sentiment scores
        print("Calculating sentiment scores...")
        sentiments = self.get_sentiment(df['news_text'].tolist())
        df = pd.concat([df, sentiments], axis=1)
        
        # Normalize features
        feature_cols = [col for col in df.columns if col not in 
                       ['datadate', 'cusip', 'ticker', 'ret', 'news_text', 'pos', 'neg', 'neu']]
        df = self.normalize_features(df, feature_cols)
        
        # Save processed data if requested
        if save_processed:
            print(f"Saving processed data to {processed_data_path}")
            df.to_pickle(processed_data_path)
        
        return df 