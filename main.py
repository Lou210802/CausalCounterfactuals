import os
import torch
import argparse
from ucimlrepo import fetch_ucirepo

from dataset import preprocess_adult_data
from model import Model
from train import train_model, test_model

CHECKPOINT_DIR = "checkpoints"


def main():
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    # Test Parameters
    parser.add_argument("--test", action="store_true", help="Test a model")

    # Model/File paths
    parser.add_argument("--model_name", type=str, default="model.pth", help="Filename for saved model")

    args = parser.parse_args()

    # Load Data
    print("Fetching dataset...")
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets

    X_train, X_val, X_test, y_train, y_val, y_test, input_size = preprocess_adult_data(X, y)

    # Initialize Model
    model = Model(input_size)

    # Construct full model path
    full_path = os.path.join(CHECKPOINT_DIR, args.model_name)

    if args.train:
        print(f"Starting training for {args.epochs} epochs...")
        train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size
        )

        # Save Weights
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        torch.save(model.state_dict(), full_path)
        print(f"Model weights saved successfully to {full_path}")

    if args.test:
        if not args.train:
            # Load weights if skipping training
            if os.path.exists(full_path):
                model.load_state_dict(torch.load(full_path))
                print(f"Loaded existing weights from {full_path}")
            else:
                print(f"Error: Weights not found at {full_path}. Cannot skip training.")
                return

        print("Running test evaluation...")
        test_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
