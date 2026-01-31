# metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class FeatureMeta:
    """
    Meta-information needed for distances and checks.

    - continuous_cols: list of column names treated as numeric continuous
    - categorical_cols: list of column names treated as categorical/binary (e.g., one-hot)
    - continuous_ranges: {col: (min, max)} computed from training set
    """
    continuous_cols: List[str]
    categorical_cols: List[str]
    continuous_ranges: Dict[str, Tuple[float, float]]


def build_feature_meta(
    train_X_df,
    continuous_cols: List[str],
) -> FeatureMeta:
    all_cols = list(train_X_df.columns)
    categorical_cols = [c for c in all_cols if c not in continuous_cols]

    continuous_ranges: Dict[str, Tuple[float, float]] = {}
    for c in continuous_cols:
        col = train_X_df[c].to_numpy(dtype=float)
        mn = float(np.nanmin(col))
        mx = float(np.nanmax(col))
        # Avoid division by zero in Gower
        if mx - mn < 1e-12:
            mx = mn + 1e-12
        continuous_ranges[c] = (mn, mx)

    return FeatureMeta(
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        continuous_ranges=continuous_ranges,
    )


def gower_distance_row(
    a: np.ndarray,
    b: np.ndarray,
    cols: Sequence[str],
    meta: FeatureMeta,
) -> float:
    """
    Gower distance between two rows a and b.
    a,b are arrays aligned to `cols`.
    """
    a = a.astype(float, copy=False)
    b = b.astype(float, copy=False)

    num_sum = 0.0
    denom = 0

    col_to_idx = {c: i for i, c in enumerate(cols)}

    # Continuous part
    for c in meta.continuous_cols:
        if c not in col_to_idx:
            continue
        i = col_to_idx[c]
        mn, mx = meta.continuous_ranges[c]
        num_sum += abs(a[i] - b[i]) / (mx - mn)
        denom += 1

    # Categorical/binary part (incl. one-hot columns)
    for c in meta.categorical_cols:
        if c not in col_to_idx:
            continue
        i = col_to_idx[c]
        num_sum += 0.0 if a[i] == b[i] else 1.0
        denom += 1

    return float(num_sum / denom) if denom > 0 else float("nan")


def avg_pairwise_gower(
    rows: List[np.ndarray],
    cols: Sequence[str],
    meta: FeatureMeta,
) -> Optional[float]:
    if len(rows) < 2:
        return None
    dists = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dists.append(gower_distance_row(rows[i], rows[j], cols, meta))
    return float(np.mean(dists)) if dists else None


def knn_plausibility_distance(
    cf_row: np.ndarray,
    train_matrix: np.ndarray,
    cols: Sequence[str],
    meta: FeatureMeta,
    k: int = 1,
) -> float:
    """
    Plausibility proxy: distance from cf to its k nearest neighbors in training data.
    Uses Gower distance for mixed data.
    Returns mean distance to kNN (k=1 => nearest).
    """
    if train_matrix.shape[0] == 0:
        return float("nan")

    # compute all gower distances (O(n*d) - fine for typical class project sizes)
    dists = np.empty(train_matrix.shape[0], dtype=float)
    for i in range(train_matrix.shape[0]):
        dists[i] = gower_distance_row(cf_row, train_matrix[i], cols, meta)

    k = max(1, min(k, len(dists)))
    idx = np.argpartition(dists, k - 1)[:k]
    return float(np.mean(dists[idx]))


def count_changed_features(
    orig: np.ndarray,
    cf: np.ndarray,
    eps: float = 1e-3,
) -> int:
    return int(np.sum(np.abs(orig - cf) > eps))


def immutable_violation(
        orig: np.ndarray,
        cf: np.ndarray,
        cols: Sequence[str],
        immutable_groups: List[List[str]],
        eps: float = 1e-3,
) -> Tuple[bool, List[str]]:
    """
    Checks if any immutable features were changed.

    Returns:
        is_violated (bool): True if any immutable feature changed.
        violated_columns (list): Names of the specific features that were illegally changed.
    """
    col_to_idx = {c: i for i, c in enumerate(cols)}
    violated_columns = []

    for group in immutable_groups:
        for c in group:
            if c not in col_to_idx:
                continue
            i = col_to_idx[c]
            # Check if the difference exceeds the epsilon threshold
            if abs(orig[i] - cf[i]) > eps:
                violated_columns.append(c)

    is_violated = len(violated_columns) > 0
    return is_violated, violated_columns


from typing import Tuple, List


def causal_violation_hard(
        orig: np.ndarray,
        cf: np.ndarray,
        cols: Sequence[str],
        causal_rules: Optional[List[Tuple[List[str], List[str]]]] = None,
        eps: float = 1e-3,
) -> Tuple[bool, List[str]]:
    """
    Very simple hard-rule check:
    causal_rules = list of (parents, children)
    Violation if any child changes while none of its parents changed.

    Returns:
        is_violated (bool): True if any causal rule was violated.
        violated_rules (list): A list of strings describing each broken relationship.
    """
    if not causal_rules:
        return False, []

    col_to_idx = {c: i for i, c in enumerate(cols)}
    violated_rules = []

    def group_changed(group_cols: List[str]) -> bool:
        for c in group_cols:
            if c not in col_to_idx:
                continue
            i = col_to_idx[c]
            if abs(orig[i] - cf[i]) > eps:
                return True
        return False

    for parents, children in causal_rules:
        parent_changed = group_changed(parents)
        child_changed = group_changed(children)

        # Logic: If the 'effect' (child) changed, but the 'cause' (parents) stayed the same,
        # the counterfactual is causally inconsistent according to the graph.
        if child_changed and not parent_changed:
            violated_rules.append(f"{parents} -> {children}")

    is_violated = len(violated_rules) > 0
    return is_violated, violated_rules
