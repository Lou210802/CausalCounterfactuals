import pandas as pd

def interpret_sample(tensor, scaler, columns, label_encoders=None):
    """
    Converts a preprocessed tensor back into a readable DataFrame row,
    handling both inverse scaling and inverse label encoding.
    """
    # Inverse Transform (Scaling)
    np_sample = tensor.detach().cpu().numpy()
    if np_sample.ndim == 1:
        np_sample = np_sample.reshape(1, -1)

    original_scale = scaler.inverse_transform(np_sample)
    df = pd.DataFrame(original_scale, columns=columns)

    # Inverse Label Encoding (Categorical Mapping)
    if label_encoders:
        for col, le in label_encoders.items():
            if col in df.columns:
                # Round and convert to int to ensure the encoder matches the original labels
                df[col] = le.inverse_transform(df[col].round().astype(int))

    return df


def compare_results(original_bin, counterfactual_bin, scaler, columns, label_encoders=None):
    """
    Displays the difference between the original and the counterfactual with readable text.
    """
    df_orig = interpret_sample(original_bin, scaler, columns, label_encoders)
    df_cf = interpret_sample(counterfactual_bin, scaler, columns, label_encoders)

    comparison = pd.concat([df_orig, df_cf], axis=0)
    comparison.index = ['Original', 'Counterfactual']

    # Only show columns that actually changed
    diff = comparison.loc['Original'] != comparison.loc['Counterfactual']
    changed_cols = diff[diff == True].index.tolist()

    return comparison[changed_cols]


def save_counterfactual_report(results, scaler, columns, label_encoders=None, filepath="counterfactual_report.txt",
                               max_rows=None):
    """
    Writes the final report using text labels instead of integer codes.
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
        lines.append(f"Index: {idx} | Success: {success} | Steps: {r['steps']} | Gower: {r['proximity_gower']:.3f}")

        if "original" in r and "counterfactual" in r:
            try:
                # Pass label_encoders here for the final translation
                comparison = compare_results(r["original"], r["counterfactual"], scaler, columns, label_encoders)
                if comparison.empty:
                    lines.append("No feature changes detected.")
                else:
                    lines.append("Changed features (Original -> Counterfactual):")
                    for col in comparison.columns:
                        orig_val = comparison.loc["Original", col]
                        cf_val = comparison.loc["Counterfactual", col]
                        lines.append(f" - {col}: {orig_val} -> {cf_val}")
            except Exception as e:
                lines.append(f"Could not compute readable diff: {e}")

        lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
