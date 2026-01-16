import pandas as pd
import torch
import torch.nn.functional as F


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
    print(f"Using device: {device}")
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
