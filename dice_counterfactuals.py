import torch
import dice_ml
import time


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


def generate_diverse_cfs(explainer, input_instance, total_cfs=4, desired_class=1):
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
            features_to_vary="all"
        )
        return dice_exp
    except Exception as e:
        print(f"Error generating CFs: {e}")
        return None


def run_dice_evaluation(model, test_df, desired_y, continuous_features, outcome_name,
                        max_scan):
    explainer = get_dice_explainer(model, test_df, continuous_features, outcome_name)
    results = []

    max_scan = len(test_df) if max_scan is None else min(max_scan, len(test_df))

    scanned = 0
    attempted = 0
    successes = 0

    for i in range(max_scan):
        scanned += 1

        # original scaled tensor
        instance_scaled = test_df.iloc[[i]].drop(columns=[outcome_name])
        orig_tensor = torch.FloatTensor(instance_scaled.values)

        orig_pred = test_df.iloc[i][outcome_name]

        # SKIP if the instance already has the desired outcome
        if int(orig_pred) == int(desired_y):
            continue

        attempted += 1

        t0 = time.perf_counter()

        cf_result = generate_diverse_cfs(explainer, instance_scaled, total_cfs=1)

        runtime_s = time.perf_counter() - t0

        if cf_result:
            try:
                successes+=1

                cf_df = cf_result.cf_examples_list[0].final_cfs_df
                cf_features_scaled = cf_df.drop(columns=[outcome_name]).iloc[[0]]
                cf_tensor = torch.FloatTensor(cf_features_scaled.values)

                # sparsity = number of changed features (in tensor space)
                eps = 1e-3
                changed = ((orig_tensor - cf_tensor).abs() > 1e-3).sum().item()

                results.append({
                    "index": i,
                    "success": True,
                    "steps": "N/A (DiCE)",  # DiCE doesn't use step counts like Adam
                    "l1_distance": torch.norm(orig_tensor - cf_tensor, p=1).item(),
                    "sparsity": changed,
                    "original": orig_tensor,
                    "counterfactual": cf_tensor,
                    "orig_pred": test_df.iloc[i][outcome_name],
                    "cf_pred": cf_df[outcome_name].iloc[0],
                    "runtime": runtime_s
                })
            except Exception as e:
                print(f"Error formatting DiCE result for index {i}: {e}")


    validity = successes / attempted if attempted > 0 else 0.0

    success_only = [r for r in results if r["success"]]

    avg_l1_success = (
        sum(r["l1_distance"] for r in success_only) / len(success_only)
        if success_only else float("nan")
    )
    avg_sparsity_success = (
        sum(r["sparsity"] for r in success_only) / len(success_only)
        if success_only else float("nan")
    )
    avg_runtime_success = (
        sum(r["runtime"] for r in success_only) / len(success_only)
        if success_only else float("nan")
    )

    avg_runtime_all = sum(r["runtime"] for r in results) / len(results) if results else 0.0

    summary = {
        "scanned": scanned,
        "attempted": attempted,
        "successes": successes,
        "validity": validity,
        "avg_l1_success": avg_l1_success,
        "avg_sparsity_success": avg_sparsity_success,
        "avg_runtime_success": avg_runtime_success,
        "avg_runtime_all": avg_runtime_all,
    }

    return results, summary
