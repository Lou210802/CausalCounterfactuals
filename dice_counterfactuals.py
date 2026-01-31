import time

import dice_ml
import numpy as np
import pandas as pd
import torch

from metrics import (
    build_feature_meta,
    gower_distance_row,
    avg_pairwise_gower,
    knn_plausibility_distance,
    immutable_violation,
    causal_violation_hard,
    count_changed_features,
)


def get_dice_explainer(model, df, continuous_features, outcome_name):
    # Identify which columns are categorical
    all_columns = [col for col in df.columns if col != outcome_name]
    categorical_features = [col for col in all_columns if col not in continuous_features]

    # Define Data with categorical info
    d = dice_ml.Data(dataframe=df,
                     continuous_features=continuous_features,
                     categorical_features=categorical_features,
                     outcome_name=outcome_name)

    # Wrap PyTorch Model
    class ProbaWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            logits = self.model(x)
            probs = torch.sigmoid(logits)
            return torch.cat([1 - probs, probs], dim=1)

    m = dice_ml.Model(model=ProbaWrapper(model), backend="PYT")

    # Use 'genetic' for better handling of categorical constraints
    exp = dice_ml.Dice(d, m, method="genetic")
    return exp


def generate_diverse_cfs(explainer, input_instance, total_cfs=4, desired_class=1, permitted_range=None):
    """
    Generates a set of diverse counterfactuals for a single instance.
    """
    try:
        dice_exp = explainer.generate_counterfactuals(
            input_instance,
            total_CFs=total_cfs,
            desired_class=desired_class,
            proximity_weight=0.5,  # Higher = closer to original
            diversity_weight=1.0,  # Higher = more diverse from each other
            features_to_vary="all",
            permitted_range=permitted_range
        )
        return dice_exp
    except Exception as e:
        print(f"Error generating CFs: {e}")
        return None


def run_dice_evaluation(model, test_df, desired_y, continuous_features, outcome_name,
                        max_scan, total_cfs=4, train_features_df=None, immutable_groups=None, causal_rules=None,
                        plausibility_k=1):
    explainer = get_dice_explainer(model, test_df, continuous_features, outcome_name)
    results = []

    if train_features_df is None:
        raise ValueError("train_features_df must be provided to compute plausibility/kNN metrics.")

    immutable_groups = immutable_groups or []

    feature_cols = [c for c in test_df.columns if c != outcome_name and c in train_features_df.columns]
    meta = build_feature_meta(train_features_df[feature_cols], continuous_features)

    # permitted ranges: restrict categorical features to seen integer codes
    permitted_range = {}
    for c in meta.categorical_cols:
        # use unique values from TRAIN, force int
        vals = train_features_df[c].dropna().unique().tolist()
        permitted_range[c] = [str(v) for v in vals]

    train_matrix = train_features_df[feature_cols].to_numpy(dtype=float)

    max_scan = len(test_df) if max_scan is None else min(max_scan, len(test_df))

    scanned = 0
    attempted = 0
    successes = 0

    for i in range(max_scan):
        scanned += 1

        instance_scaled = test_df.loc[[i], feature_cols]
        orig_tensor = torch.tensor(instance_scaled.to_numpy(copy=True), dtype=torch.float32)

        orig_pred = test_df.iloc[i][outcome_name]

        if int(orig_pred) == int(desired_y):
            continue

        attempted += 1

        instance_for_dice = instance_scaled.copy()
        for c in meta.categorical_cols:
            instance_for_dice[c] = instance_for_dice[c].astype(str)

        t0 = time.perf_counter()

        cf_result = generate_diverse_cfs(
            explainer,
            instance_for_dice,
            total_cfs=total_cfs,
            desired_class=desired_y,
            permitted_range=permitted_range
        )

        runtime_s = time.perf_counter() - t0

        if cf_result:
            try:
                successes += 1

                cf_df = cf_result.cf_examples_list[0].final_cfs_df
                num_cfs_generated = len(cf_df)
                cf_features_scaled = cf_df.drop(columns=[outcome_name])

                # DiCE may return categorical columns as strings (object dtype). Convert back to numeric.
                for c in meta.categorical_cols:
                    if c in cf_features_scaled.columns:
                        cf_features_scaled[c] = pd.to_numeric(cf_features_scaled[c], errors="coerce").fillna(0).astype(
                            float)

                orig_row = orig_tensor.numpy().reshape(-1)
                orig_cols = feature_cols

                cf_rows = [row.to_numpy(dtype=float) for _, row in cf_features_scaled.iterrows()]

                per_cf_metrics = []
                for cf_row in cf_rows:
                    prox = gower_distance_row(orig_row, cf_row, orig_cols, meta)
                    plaus = knn_plausibility_distance(cf_row, train_matrix, orig_cols, meta, k=plausibility_k)
                    immut_v, violated_cols = immutable_violation(orig_row, cf_row, orig_cols, immutable_groups)
                    caus_v, broken_rules = causal_violation_hard(orig_row, cf_row, orig_cols, causal_rules=causal_rules)
                    spars = count_changed_features(orig_row, cf_row)

                    per_cf_metrics.append({
                        "proximity_gower": prox,
                        "plausibility_knn_gower": plaus,
                        "immutable_violation": immut_v,
                        "violated_immutable_features": violated_cols,
                        "causal_violation": caus_v,
                        "violated_causal_rules": broken_rules,
                        "sparsity": spars,
                    })

                diversity_gower = avg_pairwise_gower(cf_rows, orig_cols, meta)

                # best CF = minimal Gower proximity
                best_idx = int(min(range(len(per_cf_metrics)), key=lambda k: per_cf_metrics[k]["proximity_gower"]))
                best_cf_row = cf_rows[best_idx]
                best_cf_tensor = torch.tensor(
                    best_cf_row.copy(),
                    dtype=torch.float32
                ).unsqueeze(0)

                results.append({
                    "index": i,
                    "success": True,
                    "num_cfs_generated": num_cfs_generated,
                    "steps": "N/A (DiCE)",

                    # Representative CF
                    "original": orig_tensor,
                    "counterfactual": best_cf_tensor,

                    # Per-instance metrics (based on best CF)
                    "proximity_gower": per_cf_metrics[best_idx]["proximity_gower"],
                    "plausibility_knn_gower": per_cf_metrics[best_idx]["plausibility_knn_gower"],
                    "immutable_violation": per_cf_metrics[best_idx]["immutable_violation"],
                    "violated_immutable_features": per_cf_metrics[best_idx]["violated_immutable_features"],
                    "causal_violation": per_cf_metrics[best_idx]["causal_violation"],
                    "violated_causal_rules": per_cf_metrics[best_idx]["violated_causal_rules"],
                    "sparsity": per_cf_metrics[best_idx]["sparsity"],

                    # Diversity across all CFs
                    "diversity_gower": diversity_gower,

                    # Store all CFs + per-cf metrics for later analysis
                    "counterfactuals_np": cf_rows,
                    "per_cf_metrics": per_cf_metrics,

                    "orig_pred": test_df.iloc[i][outcome_name],
                    "cf_pred": cf_df[outcome_name].iloc[best_idx] if outcome_name in cf_df.columns else None,
                    "cf_preds": cf_df[outcome_name].tolist() if outcome_name in cf_df.columns else None,

                    "runtime": runtime_s,
                })

            except Exception as e:
                print(f"Error formatting DiCE result for index {i}: {e}")

    validity = successes / attempted if attempted > 0 else 0.0

    success_only = [r for r in results if r["success"]]

    avg_runtime_all = sum(r["runtime"] for r in results) / len(results) if results else 0.0

    success_only = [r for r in results if r["success"]]

    def avg_key(key):
        vals = [r[key] for r in success_only if
                r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    print("Num feature cols:", len(feature_cols))
    print("Num continuous:", len(continuous_features))
    print("Num categorical(one-hot):", len([c for c in feature_cols if c not in continuous_features]))

    summary = {
        "scanned": scanned,
        "attempted": attempted,
        "successes": successes,
        "validity": validity,

        "avg_runtime_success": avg_key("runtime"),
        "avg_runtime_all": avg_runtime_all,

        "avg_proximity_gower": avg_key("proximity_gower"),
        "avg_plausibility_knn_gower": avg_key("plausibility_knn_gower"),
        "avg_sparsity": avg_key("sparsity"),
        "avg_diversity_gower": avg_key("diversity_gower"),

        "immutable_violation_rate": float(
            np.mean([1.0 if r["immutable_violation"] else 0.0 for r in success_only])) if success_only else float(
            "nan"),
        "causal_violation_rate": float(
            np.mean([1.0 if r["causal_violation"] else 0.0 for r in success_only])) if success_only else float("nan"),

        "avg_num_cfs_generated": float(
            np.mean([r["num_cfs_generated"] for r in success_only])) if success_only else float("nan"),
        "min_num_cfs_generated": min([r["num_cfs_generated"] for r in success_only]) if success_only else 0,
        "max_num_cfs_generated": max([r["num_cfs_generated"] for r in success_only]) if success_only else 0,
    }

    return results, summary
