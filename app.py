import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import PercentFormatter

# ======================
# Config / Defaults
# ======================
# Updated relative data paths
REL_RESULT_PATH = "csv_data/_radius_results_ALL_markets_100properties.csv"
REL_MARKET_MAP_PATH = "csv_data/F5P Active Markets.csv"
MERGED_PATH = "csv_data/ape_comps/merged_output.csv"

# Optional absolute fallbacks (kept in case someone launches outside repo root)
ABS_RESULT_PATH = "/Users/davidhuang/Documents/optimus/scripts/radius_experiment/_radius_results/_radius_results_ALL_markets.csv"
ABS_MARKET_MAP_PATH = "/Users/davidhuang/Documents/optimus/scripts/radius_experiment/F5P Active Markets.csv"

# Global figure size (tune here to control plot size and reduce scrolling)
FIG_W, FIG_H = 10, 8  # <-- Adjust these to make plots bigger/smaller

st.set_page_config(page_title="Radius Experiments", layout="centered")

st.title("Radius Comps Explorer")
st.caption(
    "Two tabs: multi-market comparison and single-market deep dive. \n\n Each market chooses random 100 properties"
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


def _bin_radius_to_1_20(r: float) -> float | None:
    """
    Round radius to nearest integer within [1, 20].
    Returns None if outside that range.
    """
    if pd.isna(r):
        return None
    r1 = round(float(r))
    if 1 <= r1 <= 20:
        return float(r1)
    return None


def make_binned_avg_tbl(avg_tbl: pd.DataFrame) -> pd.DataFrame:
    """
    From avg_tbl (market_id, radius, avg_comps_len), keep only integer radii 1–20.
    If multiple rows fall into the same bin, average them.
    """
    t = avg_tbl.copy()
    t["radius_bin"] = t["radius"].apply(_bin_radius_to_1_20)
    t = t.dropna(subset=["radius_bin"])
    out = (
        t.groupby(["market_id", "radius_bin"])["avg_comps_len"]
        .mean()
        .reset_index()
        .rename(columns={"radius_bin": "radius"})
        .sort_values(["market_id", "radius"])
    )
    return out


def market_slope_1_to_20(tbl: pd.DataFrame, market_id: int) -> float | None:
    """
    Compute slope of avg_comps_len vs radius (1–20) for a given market_id
    using least-squares linear regression.
    """
    df = tbl.query("market_id == @market_id and 1 <= radius <= 20")
    if df.empty:
        return None
    x = df["radius"].values
    y = df["avg_comps_len"].values
    if len(x) < 2:
        return None
    slope = np.polyfit(x, y, 1)[0]
    return slope


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
    # Legend logic:
    # - If more than 10 markets, suppress legend unless user explicitly enables it
    # - When shown, place legend outside to avoid covering the plot
    if drew:
        too_many = len(market_ids) > 10
        if show_legend and (not too_many):
            plt.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=7,
                ncol=1,
                frameon=False,
            )
        elif not show_legend or too_many:
            pass  # no legend

    st.pyplot(plt.gcf(), clear_figure=True)
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

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Multi-market comparison",
        "🔎 Single-market APE_local vs APE_global",
        "Single-market comps vs radius",
    ]
)

# --- Tab 1: Multi-market comparison ---
with tab1:
    st.subheader("Multi-market comparison")
    # --- Controls: Select all / Clear ---
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Select all"):
            st.session_state.tab1_selected = list(market_options)
    with c2:
        show_legend = st.checkbox(
            "Show legend",
            value=False,
            help="Toggle legend display (auto-hidden if too many markets - more than 10)",
        )

    # --- Multiselect bound to session_state ---
    selected_labels = st.multiselect(
        "Select markets",
        options=market_options,
        default=None,  # default handled by session_state
        key="tab1_selected",  # bind to session_state
        help="Compare average comps_len vs radius across multiple markets.",
    )
    run = st.button("Plot comparison")

    if run:
        sel_ids = [label_to_id[lbl] for lbl in selected_labels]
        if not sel_ids:
            st.warning("Select at least one market to compare.")
        else:
            plot_avg_lines(avg_tbl, sel_ids, name_by_id)
        binned_tbl = make_binned_avg_tbl(avg_tbl)
        rows = []
        for mid in sel_ids:
            slope = market_slope_1_to_20(binned_tbl, mid)
            avg_r5 = (
                binned_tbl.query("market_id == @mid and radius == 5")[
                    "avg_comps_len"
                ].mean()
                if not binned_tbl.query("market_id == @mid and radius == 5").empty
                else None
            )
            avg_r20 = (
                binned_tbl.query("market_id == @mid and radius == 20")[
                    "avg_comps_len"
                ].mean()
                if not binned_tbl.query("market_id == @mid and radius == 20").empty
                else None
            )
            rows.append(
                {
                    "market_id": int(mid),
                    "market_name": name_by_id.get(mid, "Unknown"),
                    "slope_r1_to_r20": slope,
                    "avg_comps_at_r5": avg_r5,
                    "avg_comps_at_r20": avg_r20,
                }
            )

        slope_df = pd.DataFrame(rows)
        st.subheader("Slope table (avg_comps_len vs radius, r=5..20)")
        st.caption(
            "Slope computed via least-squares fit on radii 5, 10, 15, 20 for each selected market."
        )
        st.dataframe(slope_df, use_container_width=True)
# --- Tab 3: Single-market deep dive ---
with tab3:
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
            "# property lines",
            min_value=1,
            value=10,
            step=1,
            help="number of property having smallest comps @ r≈15, total 100 properties",
        )
    with colC:
        hline_value = st.number_input(
            "threshold line", value=50, step=1, help="number of comps (y)"
        )

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

# --- Tab 2: APE vs Radius (folder-driven markets, avg_comps_at_r5 < 80) ---
with tab2:
    st.subheader("APE vs Radius (filtered markets)")
    st.caption("These markets have average comps @ radius = 5 miles less than 80.")

    # Path to the single merged CSV

    @st.cache_data(show_spinner=False)
    def load_merged_df(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path)

        # Normalize dtypes
        numeric_cols = [
            "radius_miles",
            "ape_local",
            "ape_global",
            "num_comps_within_radius",
            "market_avg",
            "ape_market_avg",
            "current_listing_price",
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    df = load_merged_df(MERGED_PATH)

    base_required = {
        "market_id",
        "radius_miles",
        "ape_local",
        "ape_global",
        "num_comps_within_radius",
    }
    if df.empty:
        st.warning(f"Merged CSV not found or empty: {MERGED_PATH}")
    elif not base_required.issubset(df.columns):
        st.error(
            "CSV schema mismatch in merged file.\n"
            f"Required: {sorted(list(base_required))}\n"
            f"Found: {sorted(list(df.columns))}"
        )
    else:
        # build market list from merged df
        market_ids_in_file = sorted(map(str, pd.unique(df["market_id"])))
        if not market_ids_in_file:
            st.warning("No market_ids found in merged CSV.")
        else:
            # labeling helpers
            def label_for(mid: str) -> str:
                try:
                    mid_int = int(float(mid))
                    if "name_by_id" in globals():
                        return f"{name_by_id.get(mid_int, 'Unknown')} ({mid_int})"
                except Exception:
                    pass
                return str(mid)

            label_options = [label_for(mid) for mid in market_ids_in_file]

            if st.button("Select all (merged)"):
                st.session_state.tab3_selected = label_options

            selected_labels_tab3 = st.multiselect(
                "Select markets (from merged CSV)",
                options=label_options,
                default=st.session_state.get("tab3_selected", []),
                key="tab3_selected",
                help="Markets detected from the merged CSV.",
            )

            show_comps = st.checkbox("Show # of comps (bars)", value=True)
            show_market_avg_line = st.checkbox(
                "Show APE Market-Avg (horizontal)", value=True
            )
            summary_only_selected = st.checkbox(
                "Summary: only selected markets", value=True
            )

            run_tab3 = st.button("Plot APE vs radius (from merged)")

            # === Summary table (above plots) ===
            if not df.empty:
                # effective market-avg APE per row: prefer 'ape_market_avg'; else compute from market_avg & price
                has_ape_mkt_col = (
                    "ape_market_avg" in df.columns
                    and df["ape_market_avg"].notna().any()
                )
                if has_ape_mkt_col:
                    df["ape_market_avg_effective"] = pd.to_numeric(
                        df["ape_market_avg"], errors="coerce"
                    )
                else:
                    if {"market_avg", "current_listing_price"}.issubset(df.columns):
                        tmp = df.copy()
                        tmp["market_avg"] = pd.to_numeric(
                            tmp["market_avg"], errors="coerce"
                        )
                        tmp["current_listing_price"] = pd.to_numeric(
                            tmp["current_listing_price"], errors="coerce"
                        )
                        # compute APE vs market_avg where price > 0 and both present
                        tmp["ape_market_avg_effective"] = (
                            tmp["market_avg"] - tmp["current_listing_price"]
                        ).abs() / tmp["current_listing_price"].abs()
                        tmp.loc[
                            ~(
                                (tmp["current_listing_price"] > 0)
                                & tmp["market_avg"].notna()
                                & tmp["ape_market_avg_effective"]
                                .replace([float("inf"), -float("inf")], pd.NA)
                                .notna()
                            ),
                            "ape_market_avg_effective",
                        ] = pd.NA
                        df["ape_market_avg_effective"] = pd.to_numeric(
                            tmp["ape_market_avg_effective"], errors="coerce"
                        )
                    else:
                        # no way to compute; create empty column
                        df["ape_market_avg_effective"] = pd.NA

                # Filter summary rows if desired
                def parse_mid(lbl: str) -> str:
                    if "(" in lbl and lbl.endswith(")"):
                        inside = lbl.rsplit("(", 1)[-1][:-1]
                        try:
                            return str(int(float(inside)))
                        except Exception:
                            return inside
                    return lbl

                summary_df = df.copy()
                if summary_only_selected and selected_labels_tab3:
                    selected_mids_for_summary = [
                        parse_mid(lbl) for lbl in selected_labels_tab3
                    ]
                    summary_df = summary_df[
                        summary_df["market_id"]
                        .astype(str)
                        .isin(selected_mids_for_summary)
                    ]

                # Compute per-market means
                df_summary = summary_df.groupby("market_id", as_index=False).agg(
                    ape_global=("ape_global", "mean"),
                    ape_market_avg=("ape_market_avg_effective", "mean"),
                )
                df_summary["diff"] = (
                    df_summary["ape_global"] - df_summary["ape_market_avg"]
                )

                # Optional: add market name
                if "name_by_id" in globals():

                    def market_name(mid):
                        try:
                            return name_by_id.get(int(float(mid)), "Unknown")
                        except Exception:
                            return "Unknown"

                    df_summary["market_name"] = df_summary["market_id"].apply(
                        market_name
                    )
                    cols = [
                        "market_id",
                        "market_name",
                        "ape_global",
                        "ape_market_avg",
                        "diff",
                    ]
                    df_summary = df_summary[cols]
                else:
                    cols = ["market_id", "ape_global", "ape_market_avg", "diff"]
                    df_summary = df_summary[cols]

                # Format and display
                df_summary = df_summary.sort_values("diff", ascending=False)
                st.markdown("### Market Summary (APE Global vs APE Market-Avg)")
                st.dataframe(
                    df_summary.rename(
                        columns={
                            "market_id": "Market ID",
                            "market_name": "Market Name",
                            "ape_global": "APE Global (μ)",
                            "ape_market_avg": "APE Market-Avg (μ)",
                            "diff": "Diff (Global − Market-Avg)",
                        }
                    ).round(4),
                    use_container_width=True,
                )

            # === Plots (after summary) ===
            if run_tab3:
                if not selected_labels_tab3:
                    st.info("Select at least one market.")
                else:
                    # map labels -> ids
                    def parse_mid(lbl: str) -> str:
                        if "(" in lbl and lbl.endswith(")"):
                            inside = lbl.rsplit("(", 1)[-1][:-1]
                            try:
                                return str(int(float(inside)))
                            except Exception:
                                return inside
                        return lbl

                    selected_mids = [parse_mid(lbl) for lbl in selected_labels_tab3]

                    # Aggregate once across merged df
                    agg = df.groupby(["market_id", "radius_miles"], as_index=False).agg(
                        avg_ape_local=("ape_local", "mean"),
                        avg_ape_global=("ape_global", "mean"),
                        avg_num_comps=("num_comps_within_radius", "mean"),
                    )

                    # Markets with avg comps at r=5 < 80
                    comps_r5 = (
                        agg.query("radius_miles == 5")
                        .groupby("market_id")["avg_num_comps"]
                        .mean()
                        .reset_index()
                    )
                    valid_mkts = set(
                        comps_r5.loc[comps_r5["avg_num_comps"] < 80, "market_id"]
                        .astype(str)
                        .tolist()
                    )

                    has_mkt_cols = (
                        "ape_market_avg_effective" in df.columns
                        and df["ape_market_avg_effective"].notna().any()
                    )
                    has_price = (
                        "current_listing_price" in df.columns
                    )  # kept for parity, though effective col should cover us

                    for mid in selected_mids:
                        if str(mid) not in valid_mkts:
                            st.caption(f"Skipped {mid}: avg_comps_at_r5 ≥ 80")
                            continue

                        g = agg[agg["market_id"].astype(str) == str(mid)].sort_values(
                            "radius_miles"
                        )
                        if g.empty:
                            st.caption(f"Skipped {mid}: no data after aggregation.")
                            continue

                        raw_m = df[df["market_id"].astype(str) == str(mid)]

                        # Global APE
                        ape_global_val = None
                        if not raw_m.empty and "ape_global" in raw_m.columns:
                            vals = pd.to_numeric(
                                raw_m["ape_global"], errors="coerce"
                            ).dropna()
                            if len(vals) > 0:
                                ape_global_val = float(vals.mean())

                        # Market-Avg APE (effective)
                        ape_market_val = None
                        if has_mkt_cols:
                            vals = pd.to_numeric(
                                raw_m["ape_market_avg_effective"], errors="coerce"
                            ).dropna()
                            if len(vals) > 0:
                                ape_market_val = float(vals.mean())

                        # Market display name
                        mname = None
                        try:
                            if "name_by_id" in globals():
                                mname = name_by_id.get(int(float(mid)), None)
                        except Exception:
                            pass
                        title_name = f"{mname} ({mid})" if mname else f"Market: {mid}"

                        # --- Plot ---
                        fig, ax1 = plt.subplots(figsize=(7.5, 4.5))

                        # APE Local vs radius
                        ax1.plot(
                            g["radius_miles"],
                            g["avg_ape_local"],
                            marker="o",
                            linewidth=2.0,
                            color="tab:blue",
                            label="APE Local (vs radius)",
                        )

                        # Horizontal lines
                        if pd.notna(ape_global_val):
                            ax1.axhline(
                                y=ape_global_val,
                                linestyle=(0, (5, 3)),
                                linewidth=1.8,
                                color="tab:orange",
                                label=f"APE Global (μ={ape_global_val:.2%})",
                            )
                        if show_market_avg_line and pd.notna(ape_market_val):
                            ax1.axhline(
                                y=ape_market_val,
                                linestyle=":",
                                linewidth=2.2,
                                color="tab:green",
                                label=f"APE Market-Avg (μ={ape_market_val:.2%})",
                            )

                        # Axis formatting
                        ax1.set_xlabel("Radius (miles)")
                        ax1.set_ylabel("Average APE")
                        ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

                        # Right axis with # of comps as bars (optional)
                        if show_comps:
                            ax2 = ax1.twinx()
                            ax2.bar(
                                g["radius_miles"],
                                g["avg_num_comps"],
                                alpha=0.25,
                                width=0.8,
                                color="tab:gray",
                                label="Avg # of comps",
                            )
                            ax2.set_ylabel("Average # of comps")
                            lines1, labels1 = ax1.get_legend_handles_labels()
                            lines2, labels2 = ax2.get_legend_handles_labels()
                            ax1.legend(
                                lines1 + lines2,
                                labels1 + labels2,
                                bbox_to_anchor=(1.1, 1),
                                loc="upper left",
                                fontsize=7,
                                ncol=1,
                                frameon=False,
                            )
                        else:
                            ax1.legend(loc="upper left")

                        plt.title(title_name)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
