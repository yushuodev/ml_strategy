import pandas as pd
import ta
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('ETH_label.csv')

periods = [5, 8, 13, 21, 34, 55, 89, 144, 233]
feature_columns = []

# Calculate RSI, CCI, and ADX for different periods
for period in periods:
    df[f'rsi{period}'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
    df[f'cci{period}'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=period).cci()
    df[f'adx{period}'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=period).adx()
    df[f'mfi{period}'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=period).money_flow_index()
    feature_columns.extend([f'rsi{period}', f'cci{period}', f'adx{period}', f'mfi{period}'])

macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_diff'] = macd.macd_diff()

# File path
file_path = 'expanded_features_every_row.csv'

# Feature Columns and Parameters
periods = [5, 8, 13, 21, 34, 55, 89, 144, 233]
feature_columns.extend([f'rsi{period}' for period in periods] + [f'cci{period}' for period in periods] + [f'adx{period}' for period in periods] + [f'mfi{period}' for period in periods] + ['macd', 'macd_signal', 'macd_diff'])
lags = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]  # Lags to include

# Create lagged feature columns
lagged_dfs = []
for lag in lags:
    lagged_df = df[feature_columns].shift(lag)
    lagged_df.columns = [f'{col}_{lag}' for col in feature_columns]
    lagged_dfs.append(lagged_df)

# Concatenate all lagged DataFrames
expanded_features_df = pd.concat(lagged_dfs, axis=1)
expanded_features_df.to_csv(file_path, index=False)

print('data_prepare_finished')
predictions = []
actual_labels = []
timestamps = []

training_window = 60000
prediction_range = 576
folds = 5

# Process each window
for i in tqdm(range(1000, len(expanded_features_df) - training_window - prediction_range-1000, prediction_range), desc="Processing"):
    # Create a training window
    train_window = expanded_features_df.iloc[i:i + training_window]
    train_labels = df['Label'].iloc[i:i + training_window]

    # rf_model = RandomForestClassifier(n_jobs=12)
    rf_model = AdaBoostClassifier()
    cv_scores = cross_val_score(rf_model, train_window, train_labels, cv=folds)
    print(f"Cross-Validation Scores for Window starting at index {i}: {cv_scores}")
    rf_model.fit(train_window, train_labels)

    for j in range(i + training_window, i + training_window + prediction_range):
        if j < len(expanded_features_df):
            X_pred = expanded_features_df.iloc[[j]]
            prediction = rf_model.predict(X_pred)
            predictions.append(prediction[0])
            actual_labels.append(df['Label'].iloc[j])
            timestamps.append(df['timestamp'].iloc[j])

prediction_df = pd.DataFrame({
    'Timestamp': timestamps,
    'Actual_Label': actual_labels,
    'Predicted_Label': predictions
})
prediction_df.to_csv('predictions_with_labels_and_time.csv', index=False)

prediction_df = pd.DataFrame({
    'Actual_Label': actual_labels,
    'Predicted_Label': predictions
})

# Compute the confusion matrix
cm = confusion_matrix(prediction_df['Actual_Label'], prediction_df['Predicted_Label'])

print("Confusion Matrix:")
print(cm)
