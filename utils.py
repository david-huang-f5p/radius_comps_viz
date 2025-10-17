import math
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


def safe_to_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan


def _try_kneedle(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    try:
        from kneed import KneeLocator

        kn = KneeLocator(x, y, curve="concave", direction="increasing")
        return float(kn.knee) if kn.knee is not None else None
    except Exception:
        return None


def _curvature_elbow(x: np.ndarray, y: np.ndarray) -> float:
    """
    Elbow by curvature: kappa = |y''| / (1 + y'^2)^(3/2), argmax kappa.
    Assumes x is uniform and y is smoothed.
    """
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    kappa = np.abs(d2y) / np.maximum((1.0 + dy * dy) ** 1.5, 1e-12)
    idx = int(np.argmax(kappa))
    return float(x[idx])


def _lmethod_elbow(x: np.ndarray, y: np.ndarray, min_idx: int, max_idx: int) -> float:
    """
    L-method: for each split i, fit two lines (start..i) and (i..end) by least squares,
    compute total SSE, pick i with min SSE.
    """
    best_i, best_sse = None, np.inf
    for i in range(min_idx, max_idx + 1):
        # segment 1
        x1, y1 = x[: i + 1], y[: i + 1]
        if len(x1) >= 2:
            p1 = np.polyfit(x1, y1, 1)
            y1_hat = np.polyval(p1, x1)
            sse1 = float(np.sum((y1 - y1_hat) ** 2))
        else:
            sse1 = 0.0
        # segment 2
        x2, y2 = x[i:], y[i:]
        if len(x2) >= 2:
            p2 = np.polyfit(x2, y2, 1)
            y2_hat = np.polyval(p2, x2)
            sse2 = float(np.sum((y2 - y2_hat) ** 2))
        else:
            sse2 = 0.0
        sse = sse1 + sse2
        if sse < best_sse:
            best_sse, best_i = sse, i
    return float(x[best_i])


def elbow_radius_for_market_robust(
    avg_tbl: pd.DataFrame,
    market_id: int,
    x_max_for_guard: float | None = None,  # <- allow None
    min_gap_to_tail: float = 3.0,  # <- rename from _40
    min_gap_from_start: float = 1.0,
    prefer: str = "curvature",
) -> Optional[float]:
    sub = (
        avg_tbl.loc[avg_tbl["market_id"] == market_id, ["radius", "avg_comps_len"]]
        .dropna()
        .sort_values("radius")
    )
    if len(sub) < 5:
        return _try_kneedle(sub["radius"].values, sub["avg_comps_len"].values)

    x = sub["radius"].values.astype(float)
    y = sub["avg_comps_len"].values.astype(float)

    # if not provided, guard against the true tail (e.g., 100)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max_for_guard is None:
        x_max_for_guard = x_max

    cands = {}
    try:
        cands["curvature"] = _curvature_elbow(x, y)
    except Exception:
        pass

    n = len(x)
    if n >= 10:
        left = max(2, int(0.05 * n))
        right = min(n - 3, int(0.95 * n))
        if left < right:
            try:
                cands["lmethod"] = _lmethod_elbow(x, y, left, right)
            except Exception:
                pass

    k = _try_kneedle(x, y)
    if k is not None:
        cands["kneedle"] = k

    if not cands:
        return None

    # guardrails vs **true** tail
    valid = {
        name: xr
        for name, xr in cands.items()
        if (xr - x_min) >= min_gap_from_start
        and (x_max_for_guard - xr) >= min_gap_to_tail
    }
    if not valid:

        def edge_margin(v):
            return min(v - x_min, x_max_for_guard - v)

        name = max(cands, key=lambda k: edge_margin(cands[k]))
        return float(cands[name])

    # put curvature first again
    if prefer == "auto":
        for key in ("kneedle", "curvature", "lmethod"):
            if key in valid:
                return float(valid[key])
        return float(list(valid.values())[0])

    if prefer in valid:
        return float(valid[prefer])

    for key in ("curvature", "lmethod", "kneedle"):
        if key in valid:
            return float(valid[key])
    return float(list(valid.values())[0])


def slope_elbow_to_tail(
    avg_tbl: pd.DataFrame,
    market_id: int,
    elbow_r: float,
    tail_r: float | None = None,
    min_denom: float = 3.0,
) -> float | None:
    # tail_r defaults to max available radius
    sub = avg_tbl.loc[
        avg_tbl["market_id"] == market_id, ["radius", "avg_comps_len"]
    ].dropna()
    if sub.empty or elbow_r is None or np.isnan(elbow_r):
        return None
    x_max = float(sub["radius"].max())
    tail_r = x_max if tail_r is None else float(tail_r)

    denom = tail_r - elbow_r
    if denom < min_denom:
        return None
    y1 = value_at_radius(avg_tbl, market_id, elbow_r)
    y2 = value_at_radius(avg_tbl, market_id, tail_r)
    if y1 is None or y2 is None:
        return None
    return (y2 - y1) / denom


def value_at_radius(
    avg_tbl: pd.DataFrame, market_id: int, radius: float
) -> float | None:
    """Linearly interpolate avg_comps_len at a given radius."""
    sub = (
        avg_tbl.loc[avg_tbl["market_id"] == market_id, ["radius", "avg_comps_len"]]
        .dropna()
        .sort_values("radius")
    )
    if sub.empty:
        return None

    xs = sub["radius"].values.astype(float)
    ys = sub["avg_comps_len"].values.astype(float)
    if radius <= xs.min():
        return float(ys[0])
    if radius >= xs.max():
        return float(ys[-1])
    return float(np.interp(radius, xs, ys))


def slope_between_radii(
    avg_tbl: pd.DataFrame, market_id: int, r1: float, r2: float
) -> float | None:
    if r1 == r2:
        return None
    y1 = value_at_radius(avg_tbl, market_id, r1)
    y2 = value_at_radius(avg_tbl, market_id, r2)
    if y1 is None or y2 is None:
        return None
    return (y2 - y1) / (r2 - r1)


def elbow_quality_checks(
    avg_tbl: pd.DataFrame,
    market_id: int,
    elbow_r: float | None,
    slope_e_to_tail: float | None,
    *,
    tail_r: float | None = None,
    abs_slope_cap: float = 10.0,
    rel_uplift_cap: float = 0.30,
    compare_global: bool = True,
    global_tol: float = 0.0,
) -> tuple[bool, dict]:
    diags = {
        "elbow_r": elbow_r,
        "slope_e_to_tail": slope_e_to_tail,
        "global_slope_1_to_tail": None,
        "y_elbow": None,
        "y_tail": None,
        "uplift_ratio": None,
    }
    if elbow_r is None or slope_e_to_tail is None:
        return False, diags

    sub = avg_tbl.loc[
        avg_tbl["market_id"] == market_id, ["radius", "avg_comps_len"]
    ].dropna()
    if sub.empty:
        return False, {**diags, "reason": "no_data"}
    x_max = float(sub["radius"].max())
    tail_r = x_max if tail_r is None else float(tail_r)

    # 1) global flatter?
    if compare_global:
        global_slope = slope_between_radii(avg_tbl, market_id, 1.0, tail_r)
        diags["global_slope_1_to_tail"] = global_slope
        if global_slope is None:
            return False, {**diags, "reason": "no_global_slope"}
        if slope_e_to_tail >= (global_slope - global_tol):
            return False, {**diags, "reason": "tail_not_flatter_than_global"}

    # 2) relative uplift small?
    y_elbow = value_at_radius(avg_tbl, market_id, elbow_r)
    y_tail = value_at_radius(avg_tbl, market_id, tail_r)
    diags["y_elbow"], diags["y_tail"] = y_elbow, y_tail
    if y_elbow is None or y_tail is None or y_elbow <= 0:
        return False, {**diags, "reason": "missing_values"}
    uplift_ratio = (y_tail - y_elbow) / max(y_elbow, 1e-9)
    diags["uplift_ratio"] = uplift_ratio
    if uplift_ratio > rel_uplift_cap:
        return False, {**diags, "reason": "rel_uplift_cap"}

    # 3) optional absolute cap
    if slope_e_to_tail > abs_slope_cap:
        return False, {**diags, "reason": "abs_slope_cap"}

    return True, diags


def build_market_options(df: pd.DataFrame, market_map: pd.DataFrame):
    name_by_id = dict(
        zip(market_map["census_cbsa_geoid"], market_map["census_cbsa_name"])
    )
    market_ids = sorted(
        {safe_to_int(x) for x in df["market_id"].unique() if not pd.isna(x)}
    )
    options = [
        f"{name_by_id.get(mid, 'Unknown')} ({mid})"
        for mid in market_ids
        if not math.isnan(mid)
    ]
    label_to_id = {
        f"{name_by_id.get(mid, 'Unknown')} ({mid})": int(mid)
        for mid in market_ids
        if not math.isnan(mid)
    }
    return options, label_to_id, name_by_id


@st.cache_data(show_spinner=False)
def avg_table_by_market_radius(df: pd.DataFrame):
    working = df.copy()
    working["radius"] = pd.to_numeric(working["radius"], errors="coerce")
    working = working.dropna(subset=["market_id", "radius", "comps_len"])
    out = (
        working.groupby(["market_id", "radius"])["comps_len"]
        .mean()
        .reset_index()
        .rename(columns={"comps_len": "avg_comps_len"})
    )
    out["market_id"] = out["market_id"].astype(int)
    return out


def nearest_rows_to_radius(
    subset: pd.DataFrame, target_radius: float = 15.0
) -> pd.DataFrame:
    sub = subset.copy()
    sub["radius"] = pd.to_numeric(sub["radius"], errors="coerce")
    sub = sub.dropna(subset=["radius", "property_idx", "comps_len"])
    sub["_absdiff"] = (sub["radius"] - target_radius).abs()
    idxmin = sub.groupby("property_idx")["_absdiff"].idxmin()
    nearest = sub.loc[idxmin].drop(columns=["_absdiff"])
    return nearest
