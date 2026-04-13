from pathlib import Path

"""
═══════════════════════════════════════════════════════════════════════
 Narrative Visualisation Suite  —  Spanish Energy Grid Forecasting
═══════════════════════════════════════════════════════════════════════
 Generates 7 sequential, publication-ready matplotlib plots that walk
 the reader through the scientific narrative of the ablation study.
 
 Version 2.0:
  - Benchmark refactoring: Floor, Ceiling, TSO are now vertical dashed lines
    and shaded target regions across ALL plots.
  - Plot 1 re-invented as empty evaluation space.
  - Bug fixes in lookup logic for h=1 models (Plots 6, 7).
  - Explicit distinction between "Trees" and "Tabular Neural" (Plot 3).

 Usage (inside the Jupyter Notebook):
     generate_narrative_plots(ALL_RESULTS)      # pass your global list

 Requirements: matplotlib (no Plotly).
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ═══════════════════════════════════════════════════════════════════
# UNIFIED COLOUR PALETTE — consistent across all plots
# ═══════════════════════════════════════════════════════════════════
PALETTE = {
    "floor":          "#D35400",   # Burnt orange  — Seasonal Naïve
    "tso":            "#C0392B",   # Crimson       — TSO Official
    "ceiling":        "#8E44AD",   # Amethyst      — Leakage Ceiling
    "univariate":     "#2980B9",   # Sapphire blue — N-HiTS Univariate
    "statistical":    "#27AE60",   # Emerald       — Prophet
    "tabular":        "#34495E",   # Dark grey     — MLP, TabNet
    "trees":          "#F39C12",   # Orange        — LightGBM
    "deep_sequence":  "#E74C3C",   # Red           — TFT, N-HiTS Multi
    "sandbox_a":      "#E67E22",   # Carrot        — Modeling an Edge
    "intra_uni":      "#3498DB",   # Light Blue    — h=1 Univariate
    "intra_multi":    "#16A085",   # Teal          — h=1 Multivariate
}

BG_COLOR = "#FAFAFA"
GRID_COLOR = "#E0E0E0"
ANNOTATION_FONTSIZE = 10
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12

# ═══════════════════════════════════════════════════════════════════
# HELPERS: model lookup and safe evaluation
# ═══════════════════════════════════════════════════════════════════
def _find_model(df, *search_terms, exact=None):
    if exact is not None:
        match = df[df["Model"] == exact]
        if not match.empty:
            return match.iloc[0]
    for _, row in df.iterrows():
        name_lower = row["Model"].lower()
        if all(t.lower() in name_lower for t in search_terms):
            return row
    return None

def _first_valid(*models):
    for m in models:
        if m is not None:
            return m
    return None

def _bar_color_annotate(ax, bars, values, fmt="{:,.0f}", fontsize=ANNOTATION_FONTSIZE, pad=0.015, color="#333333"):
    x_max = ax.get_xlim()[1]
    for bar, val in zip(bars, values):
        text = fmt.format(val)
        x_pos = bar.get_width() + x_max * pad
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                text, va="center", ha="left",
                fontsize=fontsize, fontweight="bold", color=color)

def _style_ax(ax, xlabel="MAE (MW)", title=""):
    ax.set_facecolor(BG_COLOR)
    ax.figure.set_facecolor("white")
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, fontweight="bold", labelpad=8)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=14, loc="left")
    ax.grid(axis="both", alpha=0.35, color=GRID_COLOR, linewidth=0.7) # Grid both axes for benchmark context
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# ═══════════════════════════════════════════════════════════════════
# COMMON: Apply standard benchmarks and shaded target region
# ═══════════════════════════════════════════════════════════════════
def _apply_fixed_benchmarks(ax, df, shade_target=True):
    floor   = _first_valid(_find_model(df, "Naïve"), _find_model(df, "Naive"))
    tso     = _first_valid(_find_model(df, "TSO", "Ceiling"), _find_model(df, "TSO"))
    ceiling = _first_valid(_find_model(df, "God Mode"), _find_model(df, "Leakage"))

    # Add Lines
    if floor is not None:
        ax.axvline(floor["MAE (MW)"], color=PALETTE["floor"], linestyle="--", linewidth=1.6, alpha=0.9, label="Naïve Floor", zorder=5)
    if ceiling is not None:
        ax.axvline(ceiling["MAE (MW)"], color=PALETTE["ceiling"], linestyle="--", linewidth=1.6, alpha=0.9, label="Leakage Ceiling", zorder=5)
    if tso is not None:
        ax.axvline(tso["MAE (MW)"], color=PALETTE["tso"], linestyle="-", linewidth=1.8, alpha=0.95, label="TSO Official", zorder=5)

    # Shade Target Zone (Highlight the desired operational range)
    if shade_target and floor is not None and ceiling is not None:
        left  = min(ceiling["MAE (MW)"], floor["MAE (MW)"])
        right = max(ceiling["MAE (MW)"], floor["MAE (MW)"])
        ax.axvspan(left, right, facecolor=PALETTE["statistical"], alpha=0.1) # Functionally colored green for acceptance

# ═══════════════════════════════════════════════════════════════════
# PLOT 1  ·  Establishing the Canvas (Bounds as lines/regions)
# ═══════════════════════════════════════════════════════════════════
def plot1_bounds(df, save_path=None):
    floor   = _first_valid(_find_model(df, "Naïve"), _find_model(df, "Naive"))
    ceiling = _first_valid(_find_model(df, "God Mode"), _find_model(df, "Leakage"))
    tso = _first_valid(_find_model(df, "TSO", "Ceiling"), _find_model(df, "TSO"))

    # If bounds aren't there, we can't establish the canvas
    if floor is None or ceiling is None: return

    fig, ax = plt.subplots(figsize=(12, 3.5))
    
    # Empty canvas, just styled
    ax.set_ylim(0, 1)
    ax.set_yticks([]) 
    
    # Apply benchmarks WITH shading (establishing the highlighted evaluation space)
    _apply_fixed_benchmarks(ax, df, shade_target=True)

    # Dynamic X-Limit
    ax.set_xlim(0, floor["MAE (MW)"] * 1.1)

    # Specific Annotations for Bounds (since they aren't bars)
    for model_row, label, color in [
        (floor,   "The Floor (Seasonal Naïve 168h)", PALETTE["floor"]),
        (ceiling, "The Ceiling (Leakage Physics / Linear Reg)", PALETTE["ceiling"]),
        (tso,     "The Target (TSO Official Forecast)", PALETTE["tso"]),  # ✅ NEW
    ]:
        if model_row is not None:
            val = model_row["MAE (MW)"]
            ax.annotate(
                f"{label}\n{val:,.0f} MW",
                xy=(val, 0.5),
                xytext=(val, 0.5),
                ha='center',
                va='center',
                fontsize=ANNOTATION_FONTSIZE,
                color=color,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#FFFFFF",
                    edgecolor="#CCCCCC",
                    alpha=0.85
                )
            )
    
    _style_ax(ax, title="Plot 1 · Establishing the Evaluation Space — Bounds as Lines & Regions")
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# PLOT 2  ·  The Pure Signal (Univariate h=24)
# ═══════════════════════════════════════════════════════════════════
def plot2_univariate(df, save_path=None):
    uni_candidates = [row for _, row in df.iterrows() if "day-ahead" in row.get("Horizon", "").lower() 
                      and not any(k in row["Model"].lower() for k in ["naïve", "naive", "tso"]) 
                      and any(k in row["Model"].lower() for k in ["univariate", "sarima"])]

    if not uni_candidates: return

    uni_candidates.sort(key=lambda r: r["MAE (MW)"], reverse=True)
    labels = [m["Model"] for m in uni_candidates]
    maes   = [m["MAE (MW)"] for m in uni_candidates]

    fig, ax = plt.subplots(figsize=(12, max(3.0, len(uni_candidates) * 0.8 + 1.5)))
    bars = ax.barh(range(len(uni_candidates)), maes, color=PALETTE["univariate"], edgecolor="white", linewidth=1.2, height=0.55)
    ax.set_yticks(range(len(uni_candidates)))
    ax.set_yticklabels(labels, fontsize=11)
    
    # Safely pad the x-axis to fit text
    # knowledge retrieval: SARIMA often has large text annotation. I must provide enough padding.
    ax.set_xlim(0, max(maes) * 1.18)
    _bar_color_annotate(ax, bars, maes)

    # Apply benchmarks (lines/shading highlight the target zone)
    _apply_fixed_benchmarks(ax, df, shade_target=True)

    legend_items = [Patch(facecolor=PALETTE["univariate"], label="Univariate Models (h=24)")]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9, framealpha=0.92, edgecolor="#CCCCCC")
    _style_ax(ax, title="Plot 2 · The Pure Signal — Univariate Autoregressive Models (h=24)")
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# PLOT 3  ·  Day-Ahead Multivariate Battle (h=24)
# ═══════════════════════════════════════════════════════════════════
def plot3_multivariate_battle(df, save_path=None):
    # Functional classification lookup table
    # knowledge retrieval: TabNet and MLP should be classified as Tabular Neural. LightGBM as trees. Group them by functional family.
    families = {
        "prophet":   ("Statistical",     PALETTE["statistical"]),
        "mlp":       ("Tabular Neural",  PALETTE["tabular"]),
        "tabnet":    ("Tabular Neural",  PALETTE["tabular"]),
        "lightgbm":  ("Boosted Trees",   PALETTE["trees"]),
        "tft":       ("Deep Sequence",   PALETTE["deep_sequence"]),
        "n-hits":    ("Deep Sequence",   PALETTE["deep_sequence"]),
    }

    multivar = []
    for _, row in df.iterrows():
        if "day-ahead" not in row.get("Horizon", "").lower(): continue
        name_l = row["Model"].lower()
        if "initial" in name_l: continue
        if any(kw in name_l for kw in ["naïve", "naive", "tso", "leakage", "univariate", "sarima", "sandbox", "god mode"]): continue
        multivar.append(row)

    if not multivar: return

    multivar.sort(key=lambda r: r["MAE (MW)"], reverse=True)
    labels = [m["Model"].replace(" Improved", "") for m in multivar]
    maes   = [m["MAE (MW)"] for m in multivar]

    # Assign functional colors functionally
    colors = []
    for m in multivar:
        assigned = "#333333" # Fallback functional: Default tabular
        for keyword, (_, color) in families.items():
            if keyword in m["Model"].lower():
                assigned = color
                break
        colors.append(assigned)

    fig, ax = plt.subplots(figsize=(13, max(4, len(multivar) * 0.6 + 2)))
    bars = ax.barh(range(len(multivar)), maes, color=colors, edgecolor="white", linewidth=1.2, height=0.6)
    ax.set_yticks(range(len(multivar)))
    ax.set_yticklabels(labels, fontsize=11)
    
    ax.set_xlim(0, max(maes) * 1.18)
    _bar_color_annotate(ax, bars, maes)

    # Apply benchmarks WITH shading (highlighting the target region)
    _apply_fixed_benchmarks(ax, df, shade_target=True)

    legend_items = [
        Patch(facecolor=PALETTE["statistical"], label="Statistical (Prophet)"),
        Patch(facecolor=PALETTE["tabular"],     label="Tabular Neural (MLP, TabNet)"),
        Patch(facecolor=PALETTE["trees"],       label="Boosted Trees (LightGBM)"),
        Patch(facecolor=PALETTE["deep_sequence"],label="Deep Sequence (TFT, N-HiTS)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9, framealpha=0.92, edgecolor="#CCCCCC")
    _style_ax(ax, title="Plot 3 · Day-Ahead Multivariate Battle — Trees vs. Deep DL (h=24)")
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# PLOT 4  ·  The Sandbox ("Modeling an Edge")
# ═══════════════════════════════════════════════════════════════════
def plot4_sandbox(df, save_path=None):

    # ── Filter strictly for Day-Ahead (h=24) ────────────────────────
    df_24 = df[~df.get("Horizon", "").astype(str).str.contains("h=1", case=False, na=False)]
    df_24 = df_24[~df_24["Model"].astype(str).str.contains("intraday|mlops", case=False, na=False)]

    # ── Model lookup ────────────────────────────────────────────────
    honest = _first_valid(
        _find_model(df_24, "LightGBM", "Multivariate"),
        _find_model(df_24, "LightGBM", "Improved"),
        _find_model(df_24, "LightGBM")
    )

    sandbox_a = _first_valid(
        _find_model(df_24, "Sandbox A", "LightGBM"),
        _find_model(df_24, "Sandbox A")
    )

    sandbox_b = _first_valid(
        _find_model(df_24, "LightGBM", "Leakage"),
        _find_model(df_24, "Leakage", "LightGBM"),
        _find_model(df_24, "LightGBM (Leakage)")
    )

    # ── Assemble models ─────────────────────────────────────────────
    models = []
    colors = []

    if honest is not None:
        models.append(honest)
        colors.append(PALETTE["trees"])

    if sandbox_a is not None:
        models.append(sandbox_a)
        colors.append(PALETTE["sandbox_a"])

    if sandbox_b is not None:
        models.append(sandbox_b)
        colors.append(PALETTE["ceiling"])  # ✅ semantic: leakage = ceiling

    if len(models) < 1:
        return

    # ── Sort worst → best ───────────────────────────────────────────
    indices = list(range(len(models)))
    indices.sort(key=lambda i: models[i]["MAE (MW)"], reverse=True)

    models = [models[i] for i in indices]
    colors = [colors[i] for i in indices]

    labels = [m["Model"].replace(" Improved", "") for m in models]
    maes   = [m["MAE (MW)"] for m in models]

    # ── Plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, max(3.0, len(models) * 0.9 + 1.2)))

    bars = ax.barh(
        range(len(models)),
        maes,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        height=0.55
    )

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=11)

    # ── Benchmarks (LINES ONLY) ─────────────────────────────────────
    _apply_fixed_benchmarks(ax, df, shade_target=False)

    # ── Axis limits BEFORE shading (important) ──────────────────────
    floor = _first_valid(_find_model(df, "Naïve"), _find_model(df, "Naive"))
    current_xmax = max(maes)
    benchmark_xmax = floor["MAE (MW)"] if floor is not None else 0
    actual_xmax = max(current_xmax, benchmark_xmax) * 1.18
    ax.set_xlim(0, actual_xmax)

    # ── Modeling gap shading (NO GAP GUARANTEE) ─────────────────────
    tso = _find_model(df, "TSO")

    if tso is not None and sandbox_b is not None:
        x_left  = sandbox_b["MAE (MW)"]
        x_right = tso["MAE (MW)"]

        eps = 1e-6  # prevent anti-aliasing gap

        ax.axvspan(
            x_left - eps,
            x_right + eps,
            facecolor=PALETTE["intra_multi"],
            alpha=0.1,
            zorder=1
        )

    # ── Value annotations ───────────────────────────────────────────
    _bar_color_annotate(ax, bars, maes)

    # ── Legend ─────────────────────────────────────────────────────
    legend_items = []

    if honest is not None:
        legend_items.append(
            Patch(facecolor=PALETTE["trees"],
                  label="Honest (Day-Ahead Constraints)")
        )

    if sandbox_a is not None:
        legend_items.append(
            Patch(facecolor=PALETTE["sandbox_a"],
                  label="Sandbox A (Modeling an Edge + TSO Forecasts)")
        )

    if sandbox_b is not None:
        legend_items.append(
            Patch(facecolor=PALETTE["ceiling"],
                  label="Sandbox B (Real-Time Generation / Leakage)")
        )

    ax.legend(handles=legend_items,
              loc="lower right",
              fontsize=9,
              framealpha=0.92,
              edgecolor="#CCCCCC")

    # ── Style ──────────────────────────────────────────────────────
    _style_ax(ax, title='Plot 4 · The Sandbox — Modeling an Edge with TSO Forecasts & Leakage')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

# ═══════════════════════════════════════════════════════════════════
# PLOT 5  ·  The Intraday Arena (h=1)
# ═══════════════════════════════════════════════════════════════════
def plot5_intraday(df, save_path=None):
    # knowledge retrieval: Plot 5 shows Intraday (h=1) models with frozen weights. Benchmarks are lines/ regions. Grouped tabular neural/Trees. TabNet and MLP should be classified as Tabular Neural. LightGBM as Trees. N-HiTS multi as Deep Sequence.

    # Strict check for h=1 or intraday horizon
    df_h1 = df[df.get("Horizon", "").str.contains("h=1", case=False, na=False) | df["Model"].str.contains("intraday", case=False, na=False)]
    
    intra_models = [row for _, row in df_h1.iterrows()]
    
    if not intra_models: return

    intra_models.sort(key=lambda r: r["MAE (MW)"], reverse=True)
    labels = [m["Model"].replace(" Improved", "") for m in intra_models]
    maes   = [m["MAE (MW)"] for m in intra_models]

    colors = [PALETTE["intra_uni"] if "univariate" in m["Model"].lower() else PALETTE["intra_multi"] for m in intra_models]

    fig, ax = plt.subplots(figsize=(12, max(3.0, len(intra_models) * 0.8 + 1.5)))
    bars = ax.barh(range(len(intra_models)), maes, color=colors, edgecolor="white", linewidth=1.2, height=0.55)
    ax.set_yticks(range(len(intra_models)))
    ax.set_yticklabels(labels, fontsize=11)
    
    ax.set_xlim(0, max(maes) * 1.18)
    _bar_color_annotate(ax, bars, maes)

    # Apply benchmarks (lines/shading highlight the target zone)
    _apply_fixed_benchmarks(ax, df, shade_target=True)

    legend_items = [
        Patch(facecolor=PALETTE["intra_uni"], label="Univariate (h=1)"),
        Patch(facecolor=PALETTE["intra_multi"], label="Multivariate Weather (h=1)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9, framealpha=0.92, edgecolor="#CCCCCC")
    
    _style_ax(ax, title="Plot 5 · The Intraday Arena — Frozen Weights (h=1)")
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# PLOT 6  ·  The Horizon Paradigm Flip (h=24 vs h=1)
# ═══════════════════════════════════════════════════════════════════
def plot6_horizon_flip(df, save_path=None):
    # Day-Ahead bests
    best_24_uni = _first_valid(_find_model(df, "N-HiTS", "Univariate", "168h"), _find_model(df, "N-HiTS", "168h"))
    best_24_mul = _first_valid(_find_model(df, "LightGBM", "Improved"), _find_model(df, "LightGBM", "Multivariate", "Ahead"))
    
    # knowledge retrieval: plot 6 is failing lookup. Fix lookup logic by strict horizon filtering.
    
    # Intraday bests (strict check)
    df_h1 = df[df.get("Horizon", "").str.contains("h=1", case=False, na=False) | df["Model"].str.contains("intraday", case=False, na=False)]
    
    best_1_uni  = _first_valid(_find_model(df_h1, "N-HiTS", "Univariate"), _find_model(df_h1, "Univariate"))
    best_1_mul  = _first_valid(_find_model(df_h1, "LightGBM", "Multivariate"), _find_model(df_h1, "Multivariate"))

    comp_models = [best_24_uni, best_24_mul, best_1_uni, best_1_mul]
    if len([m for m in comp_models if m is not None]) < 4: return

    # knowledge retrieval: title updated to "The Horizon Paradigm Flip (h=24 vs h=1)". Groups h=24 and h=1 together. Separate tabular neural/Trees.

    labels = ["Day-Ahead (h=24)", "Intraday (h=1)"]
    uni_maes = [best_24_uni["MAE (MW)"], best_1_uni["MAE (MW)"]]
    mul_maes = [best_24_mul["MAE (MW)"], best_1_mul["MAE (MW)"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, uni_maes, width, label='Best Univariate (N-HiTS Deep Sequence)', color=PALETTE["univariate"], edgecolor="white")
    rects2 = ax.bar(x + width/2, mul_maes, width, label='Best Multivariate (LightGBM Trees)', color=PALETTE["trees"], edgecolor="white")

    ax.set_ylabel('MAE (MW)', fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_title('Plot 6 · The Horizon Paradigm Flip — Univariate (Blue) vs Multivariate (Orange)', fontsize=TITLE_FONTSIZE, fontweight="bold", pad=14, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)

    # Styling
    ax.set_facecolor(BG_COLOR)
    ax.figure.set_facecolor("white")
    ax.grid(axis="y", alpha=0.35, color=GRID_COLOR, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{val:,.0f}"))

    # Annotate bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:,.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight="bold", color="#333333")

    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# PLOT 7  ·  Final Architectural Evaluation
# ═══════════════════════════════════════════════════════════════════
def plot7_staircase(df, save_path=None):

    naive    = _first_valid(_find_model(df, "Naïve"), _find_model(df, "Naive"))
    best_uni = _first_valid(_find_model(df, "N-HiTS", "Univariate"), _find_model(df, "N-HiTS", "168h"))
    best_mul = _first_valid(_find_model(df, "LightGBM", "Improved"), _find_model(df, "LightGBM"))
    
    # Strict lookup specifically for h=1 ( operational evaluation phase )
    df_h1 = df[df.get("Horizon", "").str.contains("h=1", case=False, na=False) | df["Model"].str.contains("intraday", case=False, na=False) | df["Model"].str.contains("mlops", case=False, na=False)]
    
    # FIX: Explicitly target the Daily Retrain model to grab the 265 MW result
    best_int = _first_valid(
        _find_model(df_h1, "Daily Retrain"), 
        _find_model(df_h1, "MLOps"), 
        _find_model(df_h1, "Retrain")
    )
    if best_int is None: # Fallback
        best_int = _first_valid(_find_model(df_h1, "LightGBM"))
    
    tso      = _first_valid(_find_model(df, "TSO", "Ceiling"), _find_model(df, "TSO"))
    ceiling  = _first_valid(_find_model(df, "Linear Regression", "God"), _find_model(df, "God Mode"), _find_model(df, "Leakage"))

    # We exclusively plot the machine learning architectures as bars now
    models_to_plot = [
        (best_mul, "Day-Ahead Multivariate\n(LightGBM)", PALETTE["trees"]),
        (best_uni, "Day-Ahead Univariate\n(N-HiTS)", PALETTE["univariate"]),
        (best_int, "Intraday + Continuous Learning\n(LightGBM)", PALETTE["intra_multi"]),
    ]

    valid = [(label, m["MAE (MW)"], color) for m, label, color in models_to_plot if m is not None]
    if len(valid) < 3: return

    # Sort worst to best
    valid.sort(key=lambda x: x[1], reverse=True)
    
    numbering = ["①", "②", "③"]
    labels = [f"{numbering[i]} {v[0]}" for i, v in enumerate(valid)]
    maes   = [v[1] for v in valid]
    colors = [v[2] for v in valid]

    fig, ax = plt.subplots(figsize=(13, max(4.5, len(valid) * 1.2 + 2.0))) 

    bars = ax.barh(range(len(valid)), maes, color=colors,
                edgecolor="white", linewidth=1.4, height=0.55)

    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels(labels, fontsize=11, linespacing=1.3)

    x_max = naive["MAE (MW)"] * 1.10 if naive is not None else max(maes) * 1.15
    ax.set_xlim(0, x_max)

    # Add the benchmark lines (turn OFF default shading so we can custom shade)
    _apply_fixed_benchmarks(ax, df, shade_target=False)
    
    # ─── CUSTOM ZONAL SHADING & ANNOTATIONS (Formalized) ───
    if tso is not None and ceiling is not None and naive is not None:
        v_tso = tso["MAE (MW)"]
        v_leak = ceiling["MAE (MW)"]
        v_naive = naive["MAE (MW)"]
        
        # Calculate optimal Y-height for text (top of the plot)
        y_text = len(valid) - 0.25 
        
        # Region 1: Intraday Advantage
        ax.axvspan(0, v_tso, facecolor=PALETTE["intra_multi"], alpha=0.1, zorder=0)
        ax.text(v_tso / 2, y_text, "Intraday Advantage Zone\n(Outperforming TSO Benchmark)", 
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=PALETTE["intra_multi"], alpha=0.95)
                
        # Region 2: Structural Deficit
        ax.axvspan(v_tso, v_leak, facecolor=PALETTE["ceiling"], alpha=0.08, zorder=0)
        ax.text((v_tso + v_leak) / 2, y_text, "Structural Data Deficit\n(Unobservable TSO Telemetry)", 
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=PALETTE["ceiling"], alpha=0.95)
                
        # Region 3: Standard Arena
        ax.axvspan(v_leak, v_naive, facecolor=PALETTE["tabular"], alpha=0.06, zorder=0)
        ax.text((v_leak + v_naive) / 2, y_text, "Public Data Baseline\n(Standard Model Optimization)", 
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=PALETTE["tabular"], alpha=0.85)

    # Value annotations inside/next to bars
    for bar, mae_val in zip(bars, maes):
        ax.text(bar.get_width() + x_max * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{mae_val:,.0f} MW", va="center", ha="left",
                fontsize=ANNOTATION_FONTSIZE, fontweight="bold", color="#333333")

    # Add Improvement percentages (deltas between the remaining architectural bars)
    for i in range(len(valid) - 1):
        delta = maes[i] - maes[i + 1]
        pct   = delta / maes[i] * 100
        mid_y = i + 0.5
        ax.annotate(f"−{delta:,.0f}\n({pct:.0f}%)",
                    xy=(maes[i + 1], mid_y), fontsize=8, color="#555555", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#CCCCCC", alpha=0.85), zorder=10)

    _style_ax(ax, title="Plot 7 · Final Architectural Evaluation — Predictive Performance Across Horizons")
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig("results_bar_chart.png", dpi=300, bbox_inches="tight")
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
def generate_narrative_plots(all_results, save_dir=None):
    if not all_results: return

    df = pd.DataFrame(all_results)
    if save_dir is not None: Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("━" * 60)
    print("  Plot 1 · Establishing the Evaluation Space")
    print("━" * 60)
    plot1_bounds(df, save_path=Path(save_dir) / "plot_1_bounds.png" if save_dir else None)

    print("\n" + "━" * 60)
    print("  Plot 2 · The Pure Signal (Univariate h=24)")
    print("━" * 60)
    plot2_univariate(df, save_path=Path(save_dir) / "plot_2_univariate.png" if save_dir else None)

    print("\n" + "━" * 60)
    print("  Plot 3 · Day-Ahead Multivariate Battle (h=24)")
    print("━" * 60)
    plot3_multivariate_battle(df, save_path=Path(save_dir) / "plot_3_multivariate.png" if save_dir else None)

    print("\n" + "━" * 60)
    print("  Plot 4 · The Sandbox (Modeling an Edge)")
    print("━" * 60)
    plot4_sandbox(df, save_path=Path(save_dir) / "plot_4_sandbox.png" if save_dir else None)

    print("\n" + "━" * 60)
    print("  Plot 5 · The Intraday Arena (h=1)")
    print("━" * 60)
    plot5_intraday(df, save_path=Path(save_dir) / "plot_5_intraday.png" if save_dir else None)

    print("\n" + "━" * 60)
    print("  Plot 6 · The Horizon Paradigm Flip (h=24 vs h=1)")
    print("━" * 60)
    plot6_horizon_flip(df, save_path=Path(save_dir) / "plot_6_horizon_flip.png" if save_dir else None)

    print("\n" + "━" * 60)
    print("  Plot 7 · The Grand Finale (The Evolutionary Staircase)")
    print("━" * 60)
    plot7_staircase(df, save_path=Path(save_dir) / "plot_7_staircase.png" if save_dir else None)