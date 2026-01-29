import pandas as pd
import torch
import torch.nn.functional as F
import time


def compute_distance(x, counterfactual):
    """
    Computes the distance between the original input and a counterfactual, using the L1 distance.

    Parameters:
        x (torch.tensor): The original input, of shape (1, num_features)
        counterfactual (torch.tensor): The counterfactual input. Same shape as x.

    Returns:
        distance (torch.tensor): The L1 distance between the original input x and the counterfactual input.
    """
    l1_distance = torch.norm((x - counterfactual), p=1)
    return l1_distance


def compute_output_difference(logits, desired_y):
    """
    Computes the difference between the model's output and a desired target output based on the binary cross entropy loss

    Parameters:
        output (torch.tensor): The models output for a single instance.
        desired_output (torch.tensor): The target output

    Returns:
        difference (torch.tensor): The binary cross entropy loss between output and desired_output.
    """
    # Ensure target is a float tensor on the same device as logits
    target = torch.tensor([[float(desired_y)]], device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target)


def compute_loss(x, counterfactual, logits, desired_y, lambda_reg):
    """
    Computes the loss that is minimized during counterfactual generation.
    The loss has two components:
        - the distance between the original instance x and the counterfactual instance
        - the difference between the model output and the desired output.
    The two components are combined via a weighted sum (weighted by lambda).

    Parameters:
        x (torch.tensor): The original input, of shape (1, num_features)
        counterfactual (torch.tensor): The counterfactual input. Same shape as x.
        output (torch.tensor): The models' output for a single instance of shape (1, 3).
        desired_output (torch.tensor): The target output, of same shape as the output.
        lambda_reg (float): The lambda denoting the regularization strength.

    Returns:
        difference (torch.tensor): The summed square error between output and desired_output.
    """
    distance = compute_distance(x, counterfactual)
    difference = compute_output_difference(logits, desired_y)
    return distance + lambda_reg * difference


def create_counterfactual(model, x, desired_y, feature_mask, lambda_reg=1, num_steps=1000):
    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Ensure inputs are on the same device
    x = x.to(device)
    feature_mask = feature_mask.to(device)

    counterfactual = x.clone().detach()
    counterfactual.requires_grad = True

    optimizer = torch.optim.Adam([counterfactual])

    model.eval()

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        output = model(counterfactual)

        loss = compute_loss(x, counterfactual, output, desired_y, lambda_reg)
        loss.backward()

        # Zero out gradients for features we don't want to change
        # This prevents the optimizer from updating categorical/frozen columns
        if counterfactual.grad is not None:
            counterfactual.grad.data.mul_(feature_mask)

        optimizer.step()

        # Ensure values stay within the typical scaled range
        # This prevents the features from reaching extreme, impossible values.
        with torch.no_grad():
            # For StandardScaler
            counterfactual.clamp_(-5.0, 5.0)

            counterfactual.data = x * (1 - feature_mask) + counterfactual.data * feature_mask

        # Check if the class has flipped (Threshold 0.0 for logits)
        # logit > 0 means class 1, logit < 0 means class 0
        current_pred = (output > 0.0).float().item()
        if current_pred == desired_y:
            print(f"Counterfactual found at step {step}!")
            return counterfactual.detach(), step

    return counterfactual.detach(), num_steps

def generate_counterfactuals_for_dataset(
    model,
    X,
    desired_y,
    feature_mask,
    num_attempts=20,     # how many CF attempts to run
    lambda_reg=1,
    num_steps=1000,
    max_scan=None        # how many rows to scan at most (None = full X)
):
    """
    Runs counterfactual generation on multiple instances.

    We attempt CF generation only for instances whose current prediction != desired_y.
    'num_attempts' counts how many such attempts we run.

    Returns: (results, summary)
      results: list[dict]
      summary: dict with counts + basic metrics
    """

    if num_attempts is None:
        num_attempts = float("inf")

    results = []
    model.eval()

    scanned = 0
    attempted = 0
    successes = 0

    max_scan = len(X) if max_scan is None else min(max_scan, len(X))

    for i in range(max_scan):
        if attempted >= num_attempts:
            break

        scanned += 1
        x = X[i].unsqueeze(0)

        with torch.no_grad():
            orig_logit = model(x)
            orig_pred = (orig_logit > 0.0).float().item()

        # only attempt if the instance is NOT already in the desired class
        if orig_pred == float(desired_y):
            continue

        attempted += 1

        t0 = time.perf_counter()

        cf_tensor, steps = create_counterfactual(
            model,
            x,
            desired_y=desired_y,
            feature_mask=feature_mask,
            lambda_reg=lambda_reg,
            num_steps=num_steps
        )

        runtime_s = time.perf_counter() - t0

        with torch.no_grad():
            cf_logit = model(cf_tensor)
            cf_pred = (cf_logit > 0.0).float().item()

        success = (cf_pred == float(desired_y))
        if success:
            successes += 1

        l1_dist = torch.norm(x - cf_tensor, p=1).item()

        # sparsity = number of changed features (in tensor space)
        eps = 1e-3
        changed = ((x - cf_tensor).abs() > eps).float().sum().item()

        results.append({
            "index": i,
            "orig_pred": orig_pred,
            "cf_pred": cf_pred,
            "success": success,
            "steps": steps,
            "l1_distance": l1_dist,
            "sparsity": changed,
            "original": x.detach(),
            "counterfactual": cf_tensor.detach(),
            "runtime": runtime_s,
        })

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


def interpret_sample(tensor, scaler, columns):
    """
    Converts a preprocessed tensor back into a readable DataFrame row.
    """
    # Convert tensor to numpy
    np_sample = tensor.detach().cpu().numpy()
    if np_sample.ndim == 1:
        np_sample = np_sample.reshape(1, -1)

    # Inverse Transform (Scaling)
    # This turns values like 0.12 back into '45 years old'
    original_scale = scaler.inverse_transform(np_sample)

    # Create DataFrame
    df = pd.DataFrame(original_scale, columns=columns)
    return df


def compare_results(original_bin, counterfactual_bin, scaler, columns):
    """
    Displays the difference between the original and the counterfactual.
    """
    df_orig = interpret_sample(original_bin, scaler, columns)
    df_cf = interpret_sample(counterfactual_bin, scaler, columns)

    # Combine for easy viewing
    comparison = pd.concat([df_orig, df_cf], axis=0)
    comparison.index = ['Original', 'Counterfactual']

    # Only show columns that actually changed to reduce noise
    diff = comparison.loc['Original'] != comparison.loc['Counterfactual']
    changed_cols = diff[diff == True].index.tolist()

    return comparison[changed_cols]

def save_counterfactual_report(results, scaler, columns, filepath="counterfactual_report.txt", max_rows=None):
    """
    Writes a readable report of counterfactual changes to a text file.
    Only lists changed features for each (successful) counterfactual.
    """
    lines = []
    count = 0

    for r in results:
        if max_rows is not None and count >= max_rows:
            break
        count += 1

        idx = r["index"]
        success = r["success"]
        lines.append("=" * 80)
        lines.append(f"Index: {idx} | Success: {success} | Steps: {r['steps']} | L1: {r['l1_distance']:.3f} | "
                     f"Sparsity: {r['sparsity']:.2f} | Runtime: {r.get('runtime', float('nan')):.4f}s")
        lines.append(f"Orig pred: {r.get('orig_pred')} | CF pred: {r.get('cf_pred')}")

        # Only show changes if we have tensors available
        if "original" in r and "counterfactual" in r:
            try:
                comparison = compare_results(r["original"], r["counterfactual"], scaler, columns)
                if comparison.shape[1] == 0:
                    lines.append("No feature changes detected (after inverse transform).")
                else:
                    lines.append("Changed features (Original -> Counterfactual):")
                    # Write each changed feature as a line
                    for col in comparison.columns:
                        orig_val = comparison.loc["Original", col]
                        cf_val = comparison.loc["Counterfactual", col]
                        lines.append(f" - {col}: {orig_val} -> {cf_val}")
            except Exception as e:
                lines.append(f"Could not compute readable diff: {e}")
        else:
            lines.append("No tensors stored in results to compute readable diffs.")

        lines.append("")  # blank line

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
