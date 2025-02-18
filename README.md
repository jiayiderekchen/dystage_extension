# Dystage Extension
## target 1 
build a transformer model (encoder-only) to predict ret

### data description 
data: daily_clean_ret_v2_with_news.csv, its time-series data
datadate is date, cusip and ticker are identifiers, ret is target, news_text is text data 
rest are features  

### data cleaning 
For news_text, do the following: 
1. text data preprocess  
2. use finbert to do sentiment analysis 
3. should have three columns: pos, neg, neu 
4. merge the three columns to the original dataframe 

For features, Features are normalized to the range (-1, 1) before training.

### training schedule 
1. 70 % train, 15 % validation, 15 % test 
2. Optimizer: AdamW
3. Loss function: Mean Squared Error (MSE) Loss with a mask to exclude non-existing assets from the loss computation.
4. Learning rate: 1e-5
5. Batch size: 16
6. Epoch: 10
7. Early stopping: 10
8. Save the best model
9. Save the model checkpoint
9. Save the model state dictionary
10. Save the model configuration

### model architecture 
1. number of attention heads: 16

### evaluation metrics 
1. RMSE
2. MAE
3. MAPE
4. feature importance at end of training  



