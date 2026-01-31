import argparse
import os

import pandas as pd
import torch

from causal_model import generate_causal_graph, get_learned_causal_rules, get_given_dag_spec
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
    parser.add_argument("--dice_counterfactuals", action="store_true", help="Compute counterfactuals with dice method")
    parser.add_argument("--dice_total_cfs", type=int, default=5,
                        help="Number of counterfactuals to generate per instance with DiCE (used for diversity)")
    parser.add_argument("--dice_max_scan", type=int, default=None,
                        help="How many test rows to scan at most when searching for CF candidates (default: scan full test set)")

    parser.add_argument(
        "--constraints",
        type=str,
        default="vanilla",
        choices=["vanilla", "given", "learned", "all"],
        help="Which causal constraint setting to run",
    )

    # Model/File paths
    parser.add_argument("--model_name", type=str, default="model.pth", help="Filename for saved model")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    X_train, X_val, X_test, y_train, y_val, y_test, input_size, scaler, columns, label_encoders = load_data()

    model = Model(input_size)

    full_path = os.path.join(CHECKPOINT_DIR, args.model_name)

    if args.train:
        print(f"Starting training for {args.epochs} epochs...")
        train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )

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

    if args.dice_counterfactuals:
        causal_rules = None
        if not args.train:
            if os.path.exists(full_path):
                model.load_state_dict(torch.load(full_path, map_location=device))
                model.to(device)
                model.eval()
                print(f"Loaded existing weights from {full_path}")
            else:
                print(f"Error: Weights not found at {full_path}. Please run with --train first.")
                return

        print("Computing diverse counterfactuals with DiCE...")

        test_df = pd.DataFrame(X_test.cpu().detach().numpy(), columns=columns)

        outcome_name = "target"
        with torch.no_grad():
            logits = model(X_test.to(device))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        test_df[outcome_name] = (probs > 0.5).astype(int)

        categorical_features = list(label_encoders.keys())
        continuous_features = [c for c in columns if c not in categorical_features]

        train_features_df = pd.DataFrame(X_train.cpu().detach().numpy(), columns=columns)
        for c in categorical_features:
            train_features_df[c] = train_features_df[c].astype(int)
            test_df[c] = test_df[c].astype(int)

        immutable_groups = [["age"], ["sex"], ["race"], ["native-country"]]

        feature_cols = [c for c in test_df.columns if c != outcome_name]
        dag_given = get_given_dag_spec(feature_cols)

        INTERVENTIONS_GIVEN = ["education-num", "workclass", "occupation", "hours-per-week", "marital-status"]

        def run_setting(setting: str):
            nonlocal causal_rules

            # Default = vanilla
            do_repair = False
            dag_spec = None
            rules = None
            interventions = None

            if setting == "vanilla":
                pass

            elif setting == "given":
                do_repair = True
                dag_spec = dag_given
                interventions = INTERVENTIONS_GIVEN

            elif setting == "learned":
                if causal_rules is None:
                    print("[INFO] Learned constraints selected -> generating causal graph...")
                    cg = generate_causal_graph()
                    causal_rules = get_learned_causal_rules(cg)

                do_repair = True
                dag_spec = None
                rules = causal_rules
                interventions = INTERVENTIONS_GIVEN

            else:
                raise ValueError(f"Unknown setting: {setting}")

            print(f"\n=== Running DiCE setting: {setting} ===")

            results, summary = run_dice_evaluation(
                model=model,
                test_df=test_df,
                desired_y=1,
                continuous_features=continuous_features,
                outcome_name=outcome_name,
                max_scan=args.dice_max_scan,
                total_cfs=args.dice_total_cfs,
                train_features_df=train_features_df,
                immutable_groups=(immutable_groups if setting in ["given", "learned"] else []),
                causal_rules=rules,
                plausibility_k=1,
                dag_spec=dag_spec,
                do_causal_repair=do_repair,
                intervention_features=interventions,
            )

            success_only = [r for r in results if r["success"]]
            if success_only:
                save_counterfactual_report(
                    success_only,
                    scaler,
                    columns,
                    label_encoders=label_encoders,
                    filepath=f"dice_counterfactual_report_success_{setting}.txt",
                )
                print(f"Saved dice_counterfactual_report_success_{setting}.txt")

            save_counterfactual_report(
                results,
                scaler,
                columns,
                label_encoders=label_encoders,
                filepath=f"dice_counterfactual_report_all_{setting}.txt",
                max_rows=None,
            )

            with open(f"dice_summary_{setting}.txt", "w") as f:
                f.write(f"=== DiCE Counterfactual Summary ({setting}) ===\n\n")
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

            print(f"Saved dice_summary_{setting}.txt")

        if args.constraints == "all":
            for setting in ["vanilla", "given", "learned"]:
                run_setting(setting)
        else:
            run_setting(args.constraints)


if __name__ == "__main__":
    main()
