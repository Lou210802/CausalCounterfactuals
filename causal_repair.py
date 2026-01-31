from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def topo_sort_from_parents(feature_cols: List[str], parents_map: Dict[str, List[str]]) -> List[str]:
    """Topologically sort nodes given a parent map (best-effort if cycles exist)."""
    indeg = {c: 0 for c in feature_cols}
    children = defaultdict(list)

    for child, ps in parents_map.items():
        for p in ps:
            if p in indeg and child in indeg:
                indeg[child] += 1
                children[p].append(child)

    q = deque([n for n, d in indeg.items() if d == 0])
    order = []

    while q:
        n = q.popleft()
        order.append(n)
        for ch in children.get(n, []):
            indeg[ch] -= 1
            if indeg[ch] == 0:
                q.append(ch)

    if len(order) != len(feature_cols):
        remaining = [n for n in feature_cols if n not in order]
        order.extend(remaining)

    return order


def parents_to_dag_spec(feature_cols: List[str], parents_map: Dict[str, List[str]]) -> List[Tuple[str, List[str]]]:
    """Build a stable dag_spec [(child, [parents...]), ...] from a parent map."""

    order = topo_sort_from_parents(feature_cols, parents_map)
    spec = []
    for node in order:
        ps = [p for p in parents_map.get(node, []) if p in feature_cols]
        if ps:
            spec.append((node, ps))
    return spec


def fit_structural_models(train_df, categorical_cols: List[str], dag_spec: List[Tuple[str, List[str]]]) -> Dict[
    str, Any]:
    """
    Fit one model per child: child = f(parents)
    - continuous child: DecisionTreeRegressor
    - categorical child: DecisionTreeClassifier
    """
    models: Dict[str, Any] = {}
    cat_set = set(categorical_cols)

    for child, ps in dag_spec:
        X = train_df[ps].to_numpy(dtype=float)
        y = train_df[child].to_numpy()

        if child in cat_set:
            clf = DecisionTreeClassifier(max_depth=6, random_state=0)
            clf.fit(X, y.astype(int))
            models[child] = clf
        else:
            reg = DecisionTreeRegressor(max_depth=6, random_state=0)
            reg.fit(X, y.astype(float))
            models[child] = reg

    return models


def eq(a, b, is_cat: bool, tol: float = 1e-2) -> bool:
    """Compare values with categorical casting or continuous tolerance."""
    if is_cat:
        try:
            return int(float(a)) == int(float(b))
        except Exception:
            return str(a) == str(b)
    return abs(float(a) - float(b)) <= tol


def forward_scm_from_orig(
        orig_row: np.ndarray,
        cf_row: np.ndarray,
        col_names: List[str],
        dag_spec: List[Tuple[str, List[str]]],
        struct_models: Dict[str, Any],
        categorical_cols: Optional[List[str]] = None,
        interventions: Optional[List[str]] = None,
        n_passes: int = 3,
) -> np.ndarray:
    """Forward-simulate SCM: start at orig_row, apply interventions from cf_row, then predict remaining children."""
    interventions_set = set(interventions or [])
    cat_set = set(categorical_cols or [])
    idx = {c: i for i, c in enumerate(col_names)}

    x = orig_row.copy()

    for f in interventions_set:
        if f in idx:
            x[idx[f]] = cf_row[idx[f]]

    for _ in range(n_passes):
        for child, parents in dag_spec:
            if child in interventions_set:
                continue
            if child not in idx:
                continue

            ps = [p for p in parents if p in idx]
            if not ps:
                continue

            model = struct_models.get(child)
            if model is None:
                continue

            Xp = x[[idx[p] for p in ps]].astype(float).reshape(1, -1)
            pred = model.predict(Xp)[0]

            if child in cat_set:
                x[idx[child]] = int(pred)
            else:
                x[idx[child]] = float(pred)

    return x


def violates_dag_scm(
        orig_row: np.ndarray,
        cf_row: np.ndarray,
        col_names: List[str],
        dag_spec: List[Tuple[str, List[str]]],
        struct_models: Dict[str, Any],
        interventions: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        tol: float = 1e-2,
) -> bool:
    """Return True if any non-intervention node differs from the SCM-forwarded expectation."""
    interventions_set = set(interventions or [])
    cat_set = set(categorical_cols or [])
    idx = {c: i for i, c in enumerate(col_names)}

    expected = forward_scm_from_orig(
        orig_row=orig_row,
        cf_row=cf_row,
        col_names=col_names,
        dag_spec=dag_spec,
        struct_models=struct_models,
        categorical_cols=categorical_cols,
        interventions=list(interventions_set),
        n_passes=3,
    )

    for child, _parents in dag_spec:
        if child in interventions_set:
            continue
        if child not in idx:
            continue

        is_cat = (child in cat_set)
        if not eq(cf_row[idx[child]], expected[idx[child]], is_cat=is_cat, tol=tol):
            return True

    return False
