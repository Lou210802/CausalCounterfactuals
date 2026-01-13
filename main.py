from ucimlrepo import fetch_ucirepo

from dataset import preprocess_adult_data
from model import Model
from train import train_model, test_model

import os
import torch

CHECKPOINT_DIR = "checkpoints"

def main():
    adult = fetch_ucirepo(id=2)

    X = adult.data.features
    y = adult.data.targets

    X_train, X_val, X_test, y_train, y_val, y_test, input_size = preprocess_adult_data(X, y)

    model = Model(input_size)

    train_model(model, X_train, y_train, X_val, y_val, epochs=20)

    test_model(model, X_test, y_test)

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    model_name = "model.pth"
    full_path = os.path.join(CHECKPOINT_DIR, model_name)

    torch.save(model.state_dict(), full_path)
    print(f"Model weights saved successfully to {full_path}")


if __name__ == "__main__":
    main()
