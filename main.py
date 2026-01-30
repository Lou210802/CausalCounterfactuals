import os

import pandas as pd
import torch
import argparse

from causal_model import generate_causal_graph
from counterfactuals import generate_counterfactuals_for_dataset
from counterfacutal_report import save_counterfactual_report
from dataset import load_data
from dice_counterfactuals import run_dice_evaluation
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

    # Counterfactual Parameters
    parser.add_argument("--counterfactuals", action="store_true", help="Compute counterfactuals")
    parser.add_argument("--dice_counterfactuals", action="store_true", help="Compute counterfactuals with dice method")
    parser.add_argument("--num_cfs", type=int, default=None,
                        help="Number of counterfactual attempts (default: all applicable test instances)")

    # Causal Analysis Parameters
    parser.add_argument("--causal-learn", action="store_true", help="Generate causal graph from dataset")

    # Model/File paths
    parser.add_argument("--model_name", type=str, default="model.pth", help="Filename for saved model")

    args = parser.parse_args()

    # Load Data
    X_train, X_val, X_test, y_train, y_val, y_test, input_size, scaler, columns, label_encoders = load_data()

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

    if args.counterfactuals:

        if not args.train:
            if os.path.exists(full_path):
                model.load_state_dict(torch.load(full_path, map_location="cpu"))
                print(f"Loaded existing weights from {full_path}")
            else:
                print(f"Error: Weights not found at {full_path}. Please run with --train first.")
                return

        print("Computing counterfactuals...")

        feature_mask = torch.tensor(
            [1.0 if "_" not in col else 0.0 for col in columns],
            dtype=torch.float
        )

        print("Allowed (mask=1):", [c for c, m in zip(columns, feature_mask) if m == 1])

        results, summary = generate_counterfactuals_for_dataset(
            model=model,
            X=X_test,
            desired_y=1,
            feature_mask=feature_mask,
            num_attempts=args.num_cfs,
            lambda_reg=1,
            num_steps=2000,
            max_scan=None
        )

        print(f"Scanned: {summary['scanned']}")
        print(f"Attempted: {summary['attempted']}")
        print(f"Successes: {summary['successes']}")
        print(f"Validity: {summary['validity']:.2f}")
        print(f"Avg L1 (success only): {summary['avg_l1_success']:.3f}")
        print(f"Avg Sparsity (success only): {summary['avg_sparsity_success']:.2f}")
        print(f"Avg Runtime (success only): {summary['avg_runtime_success']:.4f}s")
        print(f"Avg Runtime (all attempts): {summary['avg_runtime_all']:.4f}s")

        # Save readable report for successful counterfactuals
        success_only = [r for r in results if r["success"]]

        if success_only:
            save_counterfactual_report(
                success_only,
                scaler,
                columns,
                filepath="counterfactual_report_success.txt"
            )
            print("Saved counterfactual_report_success.txt")
        else:
            print("No successful counterfactuals to save.")

        save_counterfactual_report(results, scaler, columns, filepath="counterfactual_report_all.txt", max_rows=None)

    if args.dice_counterfactuals:
        if not args.train:
            if os.path.exists(full_path):
                model.load_state_dict(torch.load(full_path, map_location="cpu"))
                print(f"Loaded existing weights from {full_path}")
            else:
                print(f"Error: Weights not found at {full_path}. Please run with --train first.")
                return

        print("Computing diverse counterfactuals with DiCE...")

        # Prepare the DataFrame for DiCE
        # Reconstruct DF from X_test
        test_df = pd.DataFrame(X_test.cpu().detach().numpy(), columns=columns)

        # Define your outcome and continuous features
        outcome_name = "target"
        test_df[outcome_name] = (model(X_test).detach().cpu().numpy() > 0.5).astype(int)
        continuous_features = [col for col in columns if "_" not in col]

        results, summary = run_dice_evaluation(
            model=model,
            test_df=test_df,
            desired_y=1,
            continuous_features=continuous_features,
            outcome_name=outcome_name,
            max_scan=None,
        )

        print(f"Scanned: {summary['scanned']}")
        print(f"Attempted: {summary['attempted']}")
        print(f"Successes: {summary['successes']}")
        print(f"Validity: {summary['validity']:.2f}")
        print(f"Avg L1 (success only): {summary['avg_l1_success']:.3f}")
        print(f"Avg Sparsity (success only): {summary['avg_sparsity_success']:.2f}")
        print(f"Avg Runtime (success only): {summary['avg_runtime_success']:.4f}s")
        print(f"Avg Runtime (all attempts): {summary['avg_runtime_all']:.4f}s")

        # Save readable report for successful counterfactuals
        success_only = [r for r in results if r["success"]]

        if success_only:
            save_counterfactual_report(
                success_only,
                scaler,
                columns,
                filepath="dice_counterfactual_report_success.txt"
            )
            print("Saved dice_counterfactual_report_success.txt")
        else:
            print("No successful counterfactuals to save.")

        save_counterfactual_report(results, scaler, columns, filepath="dice_counterfactual_report_all.txt", max_rows=None)


    if args.causal_learn:
        print("Generating causal graph from dataset...")
        generate_causal_graph()


if __name__ == "__main__":
    main()
