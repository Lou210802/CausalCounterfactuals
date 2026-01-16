import pandas as pd
import torch
from torch import nn


def create_counterfactual(model, x, desired_y, feature_mask, lambda_reg=1, num_steps=1000):
    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Ensure inputs are on the same device
    x = x.to(device)
    feature_mask = feature_mask.to(device)

    counterfactual = x.clone().detach()
    counterfactual.requires_grad = True

    optimizer = torch.optim.Adam([counterfactual])

    # Target label needs to be a float tensor for Binary Cross Entropy
    target = torch.tensor([[float(desired_y)]], device=device)

    loss_func = nn.BCEWithLogitsLoss()

    model.eval()

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        output = model(counterfactual)

        loss_prediction = loss_func(output, target)
        loss_distance = torch.norm(counterfactual - x, p=2)

        total_loss = loss_prediction + lambda_reg * loss_distance

        total_loss.backward()

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

        # Check if the class has flipped (Threshold 0.0 for logits)
        # logit > 0 means class 1, logit < 0 means class 0
        current_pred = (output > 0.0).float().item()
        if current_pred == desired_y:
            print(f"Counterfactual found at step {step}!")
            return counterfactual.detach(), step

    return counterfactual.detach(), num_steps


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
