import argparse
import os

import pandas as pd
import torch

from causal_model import generate_causal_graph, get_learned_causal_rules, manual_causal_rules
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
    parser.add_argument("--dice_total_cfs", type=int, default=4,
                        help="Number of counterfactuals to generate per instance with DiCE (used for diversity)")
    parser.add_argument("--dice_max_scan", type=int, default=None,
                        help="How many test rows to scan at most when searching for CF candidates (default: scan full test set)")

    # Causal Analysis Parameters
    parser.add_argument("--generated_cr", action="store_true", help="Generate causal graph and use rules")
    parser.add_argument("--manual_cr", action="store_true", help="Use manual defined causal rules")

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

    causal_rules = None
    if args.generated_cr:
        print("Generating causal graph from dataset...")
        cg = generate_causal_graph()
        causal_rules = get_learned_causal_rules(cg)

    if args.manual_cr:
        causal_rules = manual_causal_rules

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
        categorical_features = list(label_encoders.keys())
        continuous_features = [c for c in columns if c not in categorical_features]

        train_features_df = pd.DataFrame(X_train.cpu().detach().numpy(), columns=columns)

        for c in categorical_features:
            train_features_df[c] = train_features_df[c].astype(int)
            test_df[c] = test_df[c].astype(int)

        immutable_groups = [
            #["age"],
            ["sex"],
            ["race"],
            ["native-country"]
        ]

        results, summary = run_dice_evaluation(
            model=model,
            test_df=test_df,
            desired_y=1,
            continuous_features=continuous_features,
            outcome_name=outcome_name,
            max_scan=args.dice_max_scan,
            total_cfs=args.dice_total_cfs,
            train_features_df=train_features_df,
            immutable_groups=immutable_groups,
            causal_rules=causal_rules,
            plausibility_k=1,
        )

        # Save readable report for successful counterfactuals
        success_only = [r for r in results if r["success"]]

        if success_only:
            save_counterfactual_report(
                success_only,
                scaler,
                columns,
                label_encoders=label_encoders,
                filepath="dice_counterfactual_report_success.txt"
            )
            print("Saved dice_counterfactual_report_success.txt")
        else:
            print("No successful counterfactuals to save.")

        save_counterfactual_report(results, scaler, columns, label_encoders=label_encoders,
                                   filepath="dice_counterfactual_report_all.txt", max_rows=None)

        with open("dice_summary.txt", "w") as f:
            f.write("=== DiCE Counterfactual Summary ===\n\n")
            f.write(f"Scanned instances: {summary['scanned']}\n")
            f.write(f"Attempted instances: {summary['attempted']}\n")
            f.write(f"Successful CFs: {summary['successes']}\n")
            f.write(f"Validity: {summary['validity']:.4f}\n\n")

            f.write("=== Distance & Sparsity ===\n")
            f.write(f"Average Gower proximity: {summary['avg_proximity_gower']:.4f}\n")
            f.write(f"Average sparsity: {summary['avg_sparsity']:.4f}\n")
            f.write(f"Average diversity (Gower): {summary['avg_diversity_gower']:.4f}\n\n")

            f.write("=== Plausibility & Constraints ===\n")
            f.write(f"Average kNN plausibility (Gower): {summary['avg_plausibility_knn_gower']:.4f}\n")
            f.write(f"Immutable violation rate: {summary['immutable_violation_rate']:.4f}\n")
            f.write(f"Causal violation rate: {summary['causal_violation_rate']:.4f}\n\n")

            f.write("=== Runtime ===\n")
            f.write(f"Average runtime (successful): {summary['avg_runtime_success']:.4f} s\n")
            f.write(f"Average runtime (all): {summary['avg_runtime_all']:.4f} s\n\n")

            f.write("=== Diversity Capacity ===\n")
            f.write(f"Avg CFs generated per instance: {summary['avg_num_cfs_generated']:.4f}\n")
            f.write(f"Min CFs generated: {summary['min_num_cfs_generated']}\n")
            f.write(f"Max CFs generated: {summary['max_num_cfs_generated']}\n")

        print("Saved dice_summary.txt")


if __name__ == "__main__":
    main()
