import os

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

DATA_CACHE_PATH = "data_cache.pt"


def load_data():
    # Check if data is available in local cache
    if os.path.exists(DATA_CACHE_PATH):
        print("Load data from cache...")
        cache = torch.load(DATA_CACHE_PATH)
        X_train = cache['X_train']
        X_val = cache['X_val']
        X_test = cache['X_test']
        y_train = cache['y_train']
        y_val = cache['y_val']
        y_test = cache['y_test']
        input_size = cache['input_size']
    else:
        print("Fetching dataset from UCI...")
        adult = fetch_ucirepo(id=2)
        X = adult.data.features
        y = adult.data.targets

        X_train, X_val, X_test, y_train, y_val, y_test, input_size = preprocess_adult_data(X, y)

        # Save dataset in cache
        print("Save data to cache...")
        torch.save({
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'input_size': input_size
        }, DATA_CACHE_PATH)

    return X_train, X_val, X_test, y_train, y_val, y_test, input_size


def preprocess_adult_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Cleans and transforms the UCI Adult Census Income dataset for PyTorch.

    This function handles missing values, encodes categorical variables into
    one-hot vectors, binarizes the target variable, and scales numerical features.
    Finally, it converts all data into PyTorch tensors.

    Args:
        X (pd.DataFrame): Feature matrix containing numerical and categorical data.
        y (pd.DataFrame): Target labels (income categories).
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before the split.

    Returns:
        tuple: A tuple containing:
            - X_train (torch.Tensor): Scaled training features.
            - X_test (torch.Tensor): Scaled testing features.
            - y_train (torch.Tensor): Binary training labels.
            - y_test (torch.Tensor): Binary testing labels.
            - input_dim (int): The number of features after one-hot encoding.
    """
    # Create copies to avoid SettingWithCopy warnings
    X = X.copy()
    y = y.copy()

    # Handle missing values marked as '?'
    X = X.replace('?', np.nan)
    # Simple imputation: fill NaN with the mode of the column
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    # Encode categorical features (One-Hot Encoding)
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Encode the Target variable
    # Convert '>50K' strings to 1 and '<=50K' to 0
    target_col = y.columns[0]
    y_encoded = y[target_col].apply(lambda x: 1 if ">50K" in str(x) else 0).values

    # Perform Train-Test-Val Split
    X_temp, X_test_raw, y_temp, y_test_np = train_test_split(
        X_encoded, y_encoded, test_size=test_size, random_state=random_state
    )

    X_train_raw, X_val_raw, y_train_np, y_val_np = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    X_train = torch.FloatTensor(X_train_scaled)
    X_val = torch.FloatTensor(X_val_scaled)
    X_test = torch.FloatTensor(X_test_scaled)

    y_train = torch.FloatTensor(y_train_np).unsqueeze(1)
    y_val = torch.FloatTensor(y_val_np).unsqueeze(1)
    y_test = torch.FloatTensor(y_test_np).unsqueeze(1)

    input_dim = X_encoded.shape[1]

    return X_train, X_val, X_test, y_train, y_val, y_test, input_dim
