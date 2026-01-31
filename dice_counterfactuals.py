import time

import dice_ml
import numpy as np
import pandas as pd
import torch

from causal_repair import (
    fit_structural_models,
    forward_scm_from_orig,
    violates_dag_scm,
    parents_to_dag_spec,
)
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
    """Build a DiCE explainer for a PyTorch classifier that outputs probabilities."""

    all_columns = [col for col in df.columns if col != outcome_name]
    categorical_features = [col for col in all_columns if col not in continuous_features]

    d = dice_ml.Data(
        dataframe=df,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        outcome_name=outcome_name,
    )

    class ProbaWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            logits = self.model(x)
            probs = torch.sigmoid(logits)
            return torch.cat([1 - probs, probs], dim=1)

    m = dice_ml.Model(model=ProbaWrapper(model), backend="PYT")

    exp = dice_ml.Dice(d, m, method="genetic")
    return exp


def generate_diverse_cfs(explainer, input_instance, total_cfs=4, desired_class=1, permitted_range=None,
                         features_to_vary="all"):
    """Generate diverse counterfactuals for one instance (DiCE genetic)."""
    try:
        dice_exp = explainer.generate_counterfactuals(
            input_instance,
            total_CFs=total_cfs,
            desired_class=desired_class,
            proximity_weight=0.5,
            diversity_weight=1.0,
            features_to_vary=features_to_vary,
            permitted_range=permitted_range,
        )
        return dice_exp
    except Exception as e:
        print(f"Error generating CFs: {e}")
        return None


def run_dice_evaluation(
        model,
        test_df,
        desired_y,
        continuous_features,
        outcome_name,
        max_scan,
        total_cfs=4,
        train_features_df=None,
        immutable_groups=None,
        causal_rules=None,
        plausibility_k=1,
        dag_spec=None,
        do_causal_repair: bool = False,
        intervention_features=None,
        scm_tol: float = 1e-2,
):
    """Run DiCE over a slice of the test set and compute distance/plausibility/constraint metrics."""
    results = []

    if train_features_df is None:
        raise ValueError("train_features_df must be provided to compute plausibility/kNN metrics.")

    immutable_groups = immutable_groups or []
    immutable_features = [f for group in immutable_groups for f in group]
    feature_cols = [c for c in test_df.columns if c != outcome_name and c in train_features_df.columns]

    meta = build_feature_meta(train_features_df[feature_cols], continuous_features)
    CAT_COLS = list(meta.categorical_cols)

    dice_test_df = test_df.copy()
    dice_train_df = train_features_df.copy()

    for c in CAT_COLS:
        if c in dice_test_df.columns:
            dice_test_df[c] = dice_test_df[c].astype(int).astype(str)
        if c in dice_train_df.columns:
            dice_train_df[c] = dice_train_df[c].astype(int).astype(str)

    explainer = get_dice_explainer(model, dice_test_df, continuous_features, outcome_name)

    # --- Prepare SCM repair ---
    intervention_features = list(intervention_features or [])
    struct_models = None

    if dag_spec is None and do_causal_repair and causal_rules:
        parents_map = {}
        for parents, children in causal_rules:
            for ch in children:
                parents_map.setdefault(ch, [])
                for p in parents:
                    parents_map[ch].append(p)

        dag_spec = parents_to_dag_spec(feature_cols, parents_map)

        if len(dag_spec) == 0:
            print("[WARN] dag_spec is empty (name mismatch?) -> SCM repair disabled")
            dag_spec = None

    if do_causal_repair and dag_spec:
        struct_models = fit_structural_models(
            train_df=train_features_df[feature_cols],
            categorical_cols=meta.categorical_cols,
            dag_spec=dag_spec,
        )

    permitted_range = {}
    for c in CAT_COLS:
        vals = dice_train_df[c].dropna().unique().tolist()
        permitted_range[c] = sorted(set(vals))

    train_matrix = train_features_df[feature_cols].to_numpy(dtype=float)
    max_scan = len(test_df) if max_scan is None else min(max_scan, len(test_df))

    scanned = 0
    attempted = 0
    successes = 0

    for i in range(max_scan):
        scanned += 1

        instance_scaled = test_df.loc[[i], feature_cols]
        orig_tensor = torch.tensor(instance_scaled.to_numpy(copy=True), dtype=torch.float32)

        orig_cols = feature_cols
        orig_row = orig_tensor.numpy().reshape(-1)

        idx_map = {c: j for j, c in enumerate(orig_cols)}
        for c in meta.categorical_cols:
            if c in idx_map:
                orig_row[idx_map[c]] = int(round(orig_row[idx_map[c]]))

        orig_pred = test_df.iloc[i][outcome_name]
        if int(orig_pred) == int(desired_y):
            continue

        attempted += 1

        instance_for_dice = dice_test_df.loc[[i], feature_cols].copy()

        permitted_range_local = dict(permitted_range)
        orig_instance = instance_scaled.iloc[0]

        for group in immutable_groups:
            for f in group:
                if f in instance_for_dice.columns:
                    v = orig_instance[f]
                    if f in CAT_COLS:
                        permitted_range_local[f] = [str(int(round(v)))]
                    else:
                        permitted_range_local[f] = [float(v), float(v)]

        features_to_vary_local = "all"

        t0 = time.perf_counter()

        cf_result = generate_diverse_cfs(
            explainer,
            instance_for_dice,
            total_cfs=total_cfs,
            desired_class=desired_y,
            permitted_range=permitted_range_local,
            features_to_vary=features_to_vary_local,
        )

        runtime_s = time.perf_counter() - t0

        if cf_result:
            try:
                successes += 1

                cf_df = cf_result.cf_examples_list[0].final_cfs_df
                num_cfs_generated = len(cf_df)

                cf_features_scaled = cf_df.drop(columns=[outcome_name])

                for c in meta.categorical_cols:
                    if c in cf_features_scaled.columns:
                        s = pd.to_numeric(cf_features_scaled[c], errors="coerce")
                        if s.isna().any():
                            orig_val = int(instance_scaled.iloc[0][c])
                            s = s.fillna(orig_val)
                        cf_features_scaled[c] = np.round(s).astype(int)

                orig_row = orig_tensor.numpy().reshape(-1)
                orig_cols = feature_cols

                raw_cf_rows = [row.to_numpy(copy=True) for _, row in cf_features_scaled.iterrows()]
                raw_cf_rows = [r.astype(float) for r in raw_cf_rows]

                idx = {c: j for j, c in enumerate(orig_cols)}

                def changed_features(orig_row, cf_row, cols, meta, tol=1e-6):
                    changed = []
                    for j, c in enumerate(cols):
                        a = orig_row[j]
                        b = cf_row[j]
                        if c in meta.categorical_cols:
                            if int(round(a)) != int(round(b)):
                                changed.append(c)
                        else:
                            if abs(float(a) - float(b)) > tol:
                                changed.append(c)
                    return changed

                cf_interventions = []
                cf_rows = []

                for raw_cf in raw_cf_rows:
                    if do_causal_repair and dag_spec and struct_models is not None:

                        dyn_changed = changed_features(orig_row, raw_cf, orig_cols, meta)
                        dyn_interventions = list(set(dyn_changed + list(intervention_features or [])))

                        dyn_interventions = [f for f in dyn_interventions if f not in immutable_features]

                        cf_input = raw_cf.copy()
                        for f in immutable_features:
                            if f in idx:
                                cf_input[idx[f]] = orig_row[idx[f]]

                        interventions_all = list(set(dyn_interventions + immutable_features))

                        repaired = forward_scm_from_orig(
                            orig_row=orig_row,
                            cf_row=cf_input,
                            col_names=orig_cols,
                            dag_spec=dag_spec,
                            struct_models=struct_models,
                            categorical_cols=meta.categorical_cols,
                            interventions=interventions_all,
                            n_passes=3,
                        )

                        cf_rows.append(repaired)
                        cf_interventions.append(interventions_all)

                    else:
                        cf_rows.append(raw_cf)
                        cf_interventions.append(list(intervention_features or []))

                per_cf_metrics = []
                for cf_row, dyn_interventions in zip(cf_rows, cf_interventions):
                    prox = gower_distance_row(orig_row, cf_row, orig_cols, meta)
                    plaus = knn_plausibility_distance(cf_row, train_matrix, orig_cols, meta, k=plausibility_k)
                    immut_v, violated_cols = immutable_violation(orig_row, cf_row, orig_cols, immutable_groups)

                    if dag_spec and struct_models is not None:

                        caus_v = violates_dag_scm(
                            orig_row=orig_row,
                            cf_row=cf_row,
                            col_names=orig_cols,
                            dag_spec=dag_spec,
                            struct_models=struct_models,
                            interventions=dyn_interventions,
                            categorical_cols=meta.categorical_cols,
                            tol=scm_tol,
                        )

                        broken_rules = []

                    else:
                        if do_causal_repair and dag_spec and struct_models is not None:
                            caus_v = violates_dag_scm(
                                orig_row=orig_row,
                                cf_row=cf_row,
                                col_names=orig_cols,
                                dag_spec=dag_spec,
                                struct_models=struct_models,
                                interventions=dyn_interventions,
                                categorical_cols=meta.categorical_cols,
                                tol=scm_tol,
                            )
                            broken_rules = []
                        else:
                            caus_v, broken_rules = causal_violation_hard(
                                orig_row,
                                cf_row,
                                orig_cols,
                                causal_rules=causal_rules,
                            )

                    spars = count_changed_features(orig_row, cf_row)

                    per_cf_metrics.append(
                        {
                            "proximity_gower": prox,
                            "plausibility_knn_gower": plaus,
                            "immutable_violation": immut_v,
                            "violated_immutable_features": violated_cols,
                            "causal_violation": caus_v,
                            "violated_causal_rules": broken_rules,
                            "sparsity": spars,
                        }
                    )

                diversity_gower = avg_pairwise_gower(cf_rows, orig_cols, meta)

                # best CF = minimal Gower proximity
                best_idx = int(min(range(len(per_cf_metrics)), key=lambda k: per_cf_metrics[k]["proximity_gower"]))
                best_cf_row = cf_rows[best_idx]
                best_cf_tensor = torch.tensor(best_cf_row.copy(), dtype=torch.float32).unsqueeze(0)

                results.append(
                    {
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
                    }
                )

            except Exception as e:
                print(f"Error formatting DiCE result for index {i}: {e}")

    validity = successes / attempted if attempted > 0 else 0.0

    success_only = [r for r in results if r["success"]]
    avg_runtime_all = sum(r["runtime"] for r in results) / len(results) if results else 0.0
    success_only = [r for r in results if r["success"]]

    def avg_key(key):
        vals = [
            r[key]
            for r in success_only
            if r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))
        ]
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
        "immutable_violation_rate": float(np.mean([1.0 if r["immutable_violation"] else 0.0 for r in success_only]))
        if success_only
        else float("nan"),
        "causal_violation_rate": float(np.mean([1.0 if r["causal_violation"] else 0.0 for r in success_only]))
        if success_only
        else float("nan"),
        "avg_num_cfs_generated": float(np.mean([r["num_cfs_generated"] for r in success_only]))
        if success_only
        else float("nan"),
        "min_num_cfs_generated": min([r["num_cfs_generated"] for r in success_only]) if success_only else 0,
        "max_num_cfs_generated": max([r["num_cfs_generated"] for r in success_only]) if success_only else 0,
    }

    return results, summary
