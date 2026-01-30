import os

from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

DATA_CACHE_PATH = "data_cache.pt"


def load_raw_dataset():
    print("Fetching dataset from UCI...")
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets
    return X, y


def load_data():
    # Check if data is available in local cache
    if os.path.exists(DATA_CACHE_PATH):
        print("Loading data and metadata from cache...")
        cache = torch.load(DATA_CACHE_PATH, weights_only=False)
        return (cache['X_train'], cache['X_val'], cache['X_test'],
                cache['y_train'], cache['y_val'], cache['y_test'],
                cache['input_size'], cache['scaler'], cache['columns'], cache['label_encoders'])
    else:
        X, y = load_raw_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test, input_dim, scaler, columns, label_encoders  = preprocess_adult_data(X, y)

        # Save dataset in cache
        print("Saving data and metadata to cache...")
        torch.save({
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'input_size': input_dim, 'scaler': scaler, 'columns': columns, 'label_encoders': label_encoders,
        }, DATA_CACHE_PATH)

        return X_train, X_val, X_test, y_train, y_val, y_test, input_dim, scaler, columns, label_encoders


def preprocess_adult_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    X = X.copy()
    y = y.copy()

    # Drop redundant features
    features_to_drop = ['fnlwgt', 'education']
    X = X.drop(columns=[col for col in features_to_drop if col in X.columns])

    # Handle missing values
    X = X.replace('?', np.nan)
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le  # Store the encoder

    # Store column names
    column_names = X.columns.tolist()

    # Encode Target
    target_col = y.columns[0]
    y_encoded = y[target_col].apply(lambda x: 1 if ">50K" in str(x) else 0).values

    # Split Data
    X_temp, X_test_raw, y_temp, y_test_np = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state
    )
    X_train_raw, X_val_raw, y_train_np, y_val_np = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    # Scale ONLY continuous features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Convert to Tensors
    X_train = torch.FloatTensor(X_train_scaled)
    X_val = torch.FloatTensor(X_val_scaled)
    X_test = torch.FloatTensor(X_test_scaled)
    y_train = torch.FloatTensor(y_train_np).unsqueeze(1)
    y_val = torch.FloatTensor(y_val_np).unsqueeze(1)
    y_test = torch.FloatTensor(y_test_np).unsqueeze(1)

    return X_train, X_val, X_test, y_train, y_val, y_test, X.shape[1], scaler, column_names, label_encoders