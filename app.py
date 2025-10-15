import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ======================
# Config / Defaults
# ======================
# Updated relative data paths
REL_RESULT_PATH = "csv_data/_radius_results_ALL_markets.csv"
REL_MARKET_MAP_PATH = "csv_data/F5P Active Markets.csv"

# Optional absolute fallbacks (kept in case someone launches outside repo root)
ABS_RESULT_PATH = "/Users/davidhuang/Documents/optimus/scripts/radius_experiment/_radius_results/_radius_results_ALL_markets.csv"
ABS_MARKET_MAP_PATH = "/Users/davidhuang/Documents/optimus/scripts/radius_experiment/F5P Active Markets.csv"

# Global figure size (tune here to control plot size and reduce scrolling)
FIG_W, FIG_H = 10, 8  # <-- Adjust these to make plots bigger/smaller

st.set_page_config(page_title="Radius Experiments", layout="wide")

st.title("Radius Comps Explorer")
st.caption(
    "Two tabs: multi-market comparison and single-market deep dive. \n\n Each market chooses random 30 properties"
)
# ======================
# Helpers
# ======================


def pick_first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def safe_to_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan


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


# --- Property detail parsing ---
# Tries to split a free-form detail string into columns, with special handling for lat/lon.
DETAIL_KV_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_ ]*?)\s*[:=]\s*([^,;|]+)")
LAT_RE = re.compile(r"\b(lat|latitude)\b", re.I)
LON_RE = re.compile(r"\b(lon|lng|longitude)\b", re.I)
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_property_detail(text: str) -> dict:
    """Parse strings like
    "lat:42.302786, lon:-83.766568, beds:3, fb:2, hb:0, sf:1853, age:75"
    into a normalized dict with keys: lat, lon, beds, fb, hb, sf, age.
    Also handles common aliases: latitude/longitude/lng.
    """
    if not isinstance(text, str) or not text.strip():
        return {}

    text = text.strip()
    out: dict[str, object] = {}

    # Split on commas, then key:value
    parts = [p for p in text.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        key = k.strip().lower().replace(" ", "_")
        val = v.strip()

        # normalize key aliases
        if key in {"latitude"}:
            key = "lat"
        elif key in {"longitude", "lng"}:
            key = "lon"

        out[key] = val

    # Keep only the fields we care about and coerce types
    keep_order = ["lat", "lon", "beds", "fb", "hb", "sf", "age"]
    cleaned: dict[str, object] = {}
    for k in keep_order:
        if k in out:
            raw = str(out[k])
            try:
                if k in ("lat", "lon"):
                    cleaned[k] = float(raw)
                else:
                    cleaned[k] = int(float(raw))
            except Exception:
                cleaned[k] = raw
    return cleaned


def plot_avg_lines(avg_tbl: pd.DataFrame, market_ids: list[int], name_by_id: dict):
    plt.figure(figsize=(FIG_W, FIG_H))
    drew = False
    for m_id in market_ids:
        sub = avg_tbl[avg_tbl["market_id"] == m_id].sort_values("radius")
        if not sub.empty:
            label = f"{name_by_id.get(m_id, 'Unknown')} ({m_id})"
            plt.plot(sub["radius"], sub["avg_comps_len"], label=label)
            drew = True
    plt.title("Average num_comps vs radius — selected markets")
    plt.xlabel("Radius")
    plt.ylabel("Average num_comps")
    plt.grid(True)
    if drew and len(market_ids) > 1:
        plt.legend()
    st.pyplot(plt.gcf())
    plt.close()


def plot_property_lines(
    df: pd.DataFrame,
    m_id: int,
    market_name: str,
    n_lines: int,
    horizontal_line: float | None,
):
    subset = df[df["market_id"] == m_id].copy()
    if subset.empty:
        st.info(f"No property data for {market_name} ({m_id}).")
        return

    nearest = nearest_rows_to_radius(subset, target_radius=15)
    if nearest.empty:
        st.info(f"Unable to rank properties for {market_name} ({m_id}).")
        return

    n = max(1, int(n_lines))
    chosen = nearest.sort_values("comps_len", ascending=True).head(n)

    plt.figure(figsize=(FIG_W, FIG_H))
    show_detailed_legend = n <= 5

    for _, row in chosen.iterrows():
        pid = row["property_idx"]
        prop_data = subset[subset["property_idx"] == pid].copy().sort_values("radius")
        if prop_data.empty:
            continue
        if show_detailed_legend:
            plt.plot(
                prop_data["radius"], prop_data["comps_len"], alpha=0.85, label=f"{pid}"
            )
        else:
            plt.plot(prop_data["radius"], prop_data["comps_len"], alpha=0.75)

    title = f"{market_name} — comps_len vs radius (smallest at radius≈15)"
    plt.title(title)
    plt.xlabel("Radius")
    plt.ylabel("Comps Length")
    plt.grid(True)

    if horizontal_line is not None:
        try:
            hval = float(horizontal_line)
            plt.axhline(y=hval, linestyle="--", color="red")
        except Exception:
            pass

    if show_detailed_legend:
        plt.legend(title="property_idx", fontsize=8)
    st.pyplot(plt.gcf())
    plt.close()

    # Details table for chosen properties at/near r=15
    if show_detailed_legend:
        base = chosen[["property_idx"]].reset_index(drop=True)
        cols_order = ["lat", "lon", "beds", "fb", "hb", "sf", "age"]
        if "property_detail" in chosen.columns:
            parsed_series = (
                chosen["property_detail"].astype(str).apply(parse_property_detail)
            )
            detail_df = (
                pd.DataFrame(list(parsed_series))
                if len(parsed_series)
                else pd.DataFrame()
            )
            if detail_df is not None and not detail_df.empty:
                # ensure all expected columns exist
                for c in cols_order:
                    if c not in detail_df.columns:
                        detail_df[c] = np.nan
                detail_df = detail_df[cols_order]
                table = pd.concat([base, detail_df], axis=1)
                st.dataframe(table, use_container_width=True)
            else:
                st.dataframe(base, use_container_width=True)
        else:
            st.dataframe(base, use_container_width=True)


# ======================
# Sidebar: data sources (paths only)
# ======================
with st.sidebar:
    st.header("Data Sources")
    st.write("The app uses relative paths by default; override below if needed.")
    result_path = pick_first_existing(REL_RESULT_PATH, ABS_RESULT_PATH)
    market_map_path = pick_first_existing(REL_MARKET_MAP_PATH, ABS_MARKET_MAP_PATH)

    result_path = st.text_input(
        "Results CSV path", value=result_path or REL_RESULT_PATH
    )
    market_map_path = st.text_input(
        "Market map CSV path", value=market_map_path or REL_MARKET_MAP_PATH
    )

# ======================
# Load Data
# ======================
try:
    df = load_csv(result_path)
    market_map = load_csv(market_map_path)
except Exception as e:
    st.error(f"Failed to load CSVs. Check paths/files.\n\n{e}")
    st.stop()

# Normalize key columns
if "market_id" in df.columns:
    df["market_id"] = pd.to_numeric(df["market_id"], errors="coerce").astype("Int64")

market_options, label_to_id, name_by_id = build_market_options(df, market_map)
avg_tbl = avg_table_by_market_radius(df)

# ======================
# Tabs
# ======================

tab1, tab2 = st.tabs(["📊 Multi-market comparison", "🔎 Single-market deep dive"])

# --- Tab 1: Multi-market comparison ---
with tab1:
    st.subheader("Multi-market comparison")

    default_labels = [
        opt for opt in market_options if any(x in opt for x in ["(12060)", "(31080)"])
    ]
    selected_labels = st.multiselect(
        "Select markets",
        options=market_options,
        default=default_labels,
        help="Compare average comps_len vs radius across multiple markets.",
    )
    run = st.button("Plot comparison")

    if run:
        sel_ids = [label_to_id[lbl] for lbl in selected_labels]
        if not sel_ids:
            st.warning("Select at least one market to compare.")
        else:
            plot_avg_lines(avg_tbl, sel_ids, name_by_id)

# --- Tab 2: Single-market deep dive ---
with tab2:
    st.subheader("Single-market deep dive")
    st.caption("When property lines <= 5, viz will pop out property details")

    selected_labels_2 = st.multiselect(
        "Select one or more markets",
        options=market_options,
        help="Shows one averaged chart for all selected markets; optionally plots property-level lines per market.",
    )

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        show_property_plots = st.checkbox("Show property lines", value=True)
    with colB:
        n_lines = st.number_input(
            "# property lines (smallest @ r≈15, total 30 properties)",
            min_value=1,
            value=10,
            step=1,
        )
    with colC:
        hline_value = st.number_input("threshold - len of comps (y)", value=50, step=1)

    if not selected_labels_2:
        st.info("Select at least one market to display.")
    else:
        sel_ids2 = [label_to_id[lbl] for lbl in selected_labels_2]
        # 1) Average lines for selected markets
        plot_avg_lines(avg_tbl, sel_ids2, name_by_id)

        # 2) Per-market property lines
        if show_property_plots:
            for mid in sel_ids2:
                mname = name_by_id.get(mid, "Unknown")
                st.markdown(f"**{mname} ({mid}) — property lines**")
                plot_property_lines(df, mid, mname, n_lines, hline_value)
