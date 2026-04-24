"""
make_figures.py — Regenerate all paper figures from Phase 1 and Phase 2 results.

Produces:
  figures/fig1_main_result.{pdf,png}          — baseline left-lean across instruments
  figures/fig2_decomposition.{pdf,png}        — asymmetric accommodation by condition
  figures/fig5_perception_mechanism.{pdf,png} — %Dem-closer heatmap by model x condition
  figures/fig6_item_framing_distribution.{pdf,png} — item-level shift scatter
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Unified model palette (ColorBrewer "Set2"-style, colorblind-friendly)
MODEL_COLORS = {
    "GPT-4o":            "#4878cf",
    "GPT-5":             "#6acc65",
    "Claude Sonnet 4.5": "#956cb4",
    "Gemini 2.5 Flash":  "#d65f5f",
    "Llama 4 Maverick":  "#ee854a",
    "DeepSeek-R1":       "#8c613c",
}

MODEL_ORDER = [
    "GPT-4o", "GPT-5", "Claude Sonnet 4.5",
    "Gemini 2.5 Flash", "Llama 4 Maverick", "DeepSeek-R1",
]

COND_COLORS = {
    "N":   "#7f7f7f",
    "CA":  "#bcbd22",
    "C3L": "#1f77b4",
    "C3R": "#d62728",
    "C1L": "#9ecae1",
    "C1R": "#fc9272",
}

COND_LABELS = {
    "N":   "Default",
    "CA":  "Academic\nauditor",
    "C1L": "CAP\nresearcher",
    "C1R": "Heritage\nresearcher",
    "C3L": "Progressive\nDemocrat",
    "C3R": "Conservative\nRepublican",
}


def save(fig, name):
    path_pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    path_png = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path_png, bbox_inches="tight", pad_inches=0.05, dpi=200)
    print(f"  -> {path_pdf}")


# ------------------------------------------------------------------
# Figure 1 — baseline left-lean across instruments
# ------------------------------------------------------------------
def fig1_main():
    print("\nFigure 1: baseline left-lean across three instruments...")

    pct = pd.read_csv(os.path.join(DATA_DIR, "phase1_pct_scores.csv"))
    typ = pd.read_csv(os.path.join(DATA_DIR, "phase1_typology_scores.csv"))
    atp_raw = pd.read_csv(os.path.join(DATA_DIR, "phase1_atp_alignment.csv"))
    atp = atp_raw.groupby("model_label").apply(
        lambda g: pd.Series({
            "pct_closer_to_dem": (g["closer_to"] == "Dem").mean() * 100,
        }), include_groups=False
    ).reset_index()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 3.4))

    LEFT_BLUE = "#2e5c8a"
    RIGHT_RED = "#b3412c"
    y = np.arange(len(MODEL_ORDER))

    def ideology_panel(ax, lookup, lim, xlabel, title, fmt="{:+.2f}"):
        vals = [lookup.get(m, np.nan) for m in MODEL_ORDER]
        colors = [RIGHT_RED if (v is not None and not np.isnan(v) and v > 0) else LEFT_BLUE for v in vals]
        ax.axvspan(-lim, 0, facecolor=LEFT_BLUE, alpha=0.05, zorder=0)
        ax.axvspan(0, lim, facecolor=RIGHT_RED, alpha=0.05, zorder=0)
        ax.barh(y, vals, color=colors, alpha=0.9, edgecolor="white",
                linewidth=0.6, height=0.7, zorder=3)
        ax.axvline(0, color="black", linewidth=0.9, zorder=2)
        ax.set_yticks(y)
        ax.invert_yaxis()
        ax.set_xlim(-lim, lim)
        ax.set_xlabel(xlabel)
        ax.set_title(title, pad=10)
        ax.grid(axis="x", alpha=0.2, linestyle=":", zorder=1)
        for yi, v in zip(y, vals):
            ax.text(v - lim*0.04 if v < 0 else v + lim*0.04, yi, fmt.format(v),
                    ha="right" if v < 0 else "left", va="center",
                    fontsize=8, fontweight="bold", color="#222")
        ax.text(-lim*0.97, -0.9, "left", fontsize=7.5, color=LEFT_BLUE,
                fontweight="bold", ha="left")
        ax.text(lim*0.97, -0.9, "right", fontsize=7.5, color=RIGHT_RED,
                fontweight="bold", ha="right")

    # Panel 1: PCT
    pct_lookup = dict(zip(pct["model_label"], pct["mean_ideo"]))
    ideology_panel(ax1, pct_lookup, 2.0,
                   "PCT ideology score\n(−2 = left · +2 = right)",
                   "Political Compass Test (62 items)")
    ax1.set_yticklabels(MODEL_ORDER)

    # Panel 2: Typology
    typ_lookup = dict(zip(typ["model_label"], typ["mean_ideo"]))
    ideology_panel(ax2, typ_lookup, 1.0,
                   "Typology ideology score\n(−1 = left · +1 = right)",
                   "Pew Political Typology (25 items)")
    ax2.set_yticklabels([])

    # Panel 3: ATP share closer to Dem
    atp_lookup = dict(zip(atp["model_label"], atp["pct_closer_to_dem"]))
    vals3 = [atp_lookup.get(m, np.nan) for m in MODEL_ORDER]
    ax3.axvspan(50, 100, facecolor=LEFT_BLUE, alpha=0.05, zorder=0)
    ax3.axvspan(0, 50, facecolor=RIGHT_RED, alpha=0.05, zorder=0)
    ax3.barh(y, vals3, color=LEFT_BLUE, alpha=0.9, edgecolor="white",
             linewidth=0.6, height=0.7, zorder=3)
    ax3.axvline(50, color="black", linewidth=0.9, linestyle="--", zorder=2)
    ax3.set_yticks(y)
    ax3.set_yticklabels([])
    ax3.invert_yaxis()
    ax3.set_xlim(0, 100)
    ax3.set_xticks([0, 25, 50, 75, 100])
    n_atp_items = atp_raw["question_id"].nunique()
    ax3.set_xlabel(f"% of {n_atp_items:,} ATP items closer\nto Democrats than Republicans")
    ax3.set_title("Pew American Trends Panel", pad=10)
    ax3.grid(axis="x", alpha=0.2, linestyle=":", zorder=1)
    for yi, v in zip(y, vals3):
        ax3.text(v + 1.5, yi, f"{v:.0f}%", ha="left", va="center",
                 fontsize=8, fontweight="bold", color="#222")
    ax3.text(2, -0.9, "more Rep-like", fontsize=7.5, color=RIGHT_RED,
             fontweight="bold", ha="left")
    ax3.text(98, -0.9, "more Dem-like", fontsize=7.5, color=LEFT_BLUE,
             fontweight="bold", ha="right")

    fig.suptitle("All six models lean left at baseline across three instruments",
                 fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_main_result")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 2 — asymmetric accommodation
# ------------------------------------------------------------------
def fig2_decomp():
    print("\nFigure 2: asymmetric accommodation...")

    shift = pd.read_csv(os.path.join(DATA_DIR, "phase2_shift_summary.csv"))
    decomp = pd.read_csv(os.path.join(DATA_DIR, "phase2_decomposition.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    y = np.arange(len(MODEL_ORDER))
    bar_h = 0.38
    c3r = dict(zip(decomp["model_label"], decomp["sycophancy_right"]))
    c3l = dict(zip(decomp["model_label"], decomp["sycophancy_left"]))

    vals_c3r = [c3r.get(m, 0) for m in MODEL_ORDER]
    vals_c3l = [c3l.get(m, 0) for m in MODEL_ORDER]

    ax1.barh(y - bar_h/2, vals_c3r, bar_h, label="Conservative Republican (C3R)",
             color="#b3412c", edgecolor="white", linewidth=0.5, zorder=3)
    ax1.barh(y + bar_h/2, vals_c3l, bar_h, label="Progressive Democrat (C3L)",
             color="#2e5c8a", edgecolor="white", linewidth=0.5, zorder=3)

    ax1.axvline(0, color="black", linewidth=0.8, zorder=2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(MODEL_ORDER)
    ax1.invert_yaxis()
    ax1.set_xlabel("Accommodation toward asker (Δ Wasserstein distance vs. baseline)")
    ax1.set_title("Rightward shift ≈ 8x larger than leftward shift", pad=8)
    ax1.legend(loc="lower right", frameon=False, fontsize=7.5,
               handlelength=1.2, handletextpad=0.5)
    ax1.grid(axis="x", alpha=0.2, linestyle=":", zorder=1)
    for yi, v in zip(y, vals_c3r):
        ax1.text(v + 0.002, yi - bar_h/2, f"{v:.02f}", fontsize=7,
                 va="center", ha="left", color="#b3412c", fontweight="bold")

    slant = dict(zip(decomp["model_label"], decomp["baseline_slant"]))
    xs = np.array([slant[m] for m in MODEL_ORDER])
    ys = np.array([c3r[m] for m in MODEL_ORDER])

    # Fit line first so points sit on top
    slope, intercept = np.polyfit(xs, ys, 1)
    xp = np.linspace(xs.min() - 0.008, xs.max() + 0.008, 50)
    ax2.plot(xp, slope * xp + intercept, "--", color="#888",
             alpha=0.7, linewidth=1.2, zorder=2)

    for m, x, yv in zip(MODEL_ORDER, xs, ys):
        ax2.scatter(x, yv, s=90, color=MODEL_COLORS[m],
                    edgecolor="white", linewidth=1.2, zorder=4)

    # Manual label placement to avoid overlaps
    label_offsets = {
        "GPT-4o":            (-10,   8),
        "GPT-5":             (-10, -12),
        "Claude Sonnet 4.5": ( 10,   2),
        "Gemini 2.5 Flash":  ( 10,   2),
        "Llama 4 Maverick":  (-10,   8),
        "DeepSeek-R1":       (-10,  -4),
    }
    label_ha = {
        "GPT-4o": "right", "GPT-5": "right",
        "Claude Sonnet 4.5": "left", "Gemini 2.5 Flash": "left",
        "Llama 4 Maverick": "right", "DeepSeek-R1": "right",
    }
    for m, x, yv in zip(MODEL_ORDER, xs, ys):
        dx, dy = label_offsets[m]
        ax2.annotate(m, (x, yv), fontsize=7.5, xytext=(dx, dy),
                     textcoords="offset points", ha=label_ha[m],
                     color="#222", fontweight="semibold")

    r = np.corrcoef(xs, ys)[0, 1]
    ax2.text(0.04, 0.96, f"r = {r:+.2f}", transform=ax2.transAxes,
             fontsize=11, va="top", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", lw=0.6))
    ax2.set_xlabel("Baseline left-slant\n(WD(Rep) − WD(Dem) at N)")
    ax2.set_ylabel("Rightward accommodation under C3R")
    ax2.set_title("Larger baseline slant predicts\nlarger right accommodation", pad=8)
    ax2.grid(alpha=0.2, linestyle=":")
    # Pad axes generously so labels don't run into edge
    xpad = (xs.max() - xs.min()) * 0.30
    ypad = (ys.max() - ys.min()) * 0.35
    ax2.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax2.set_ylim(ys.min() - ypad, ys.max() + ypad)

    fig.tight_layout()

    save(fig, "fig2_decomposition")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 5 — %Dem-closer heatmap
# ------------------------------------------------------------------
def fig5_heatmap():
    print("\nFigure 5: %Dem-closer heatmap...")

    align = pd.read_csv(os.path.join(DATA_DIR, "phase2_alignment_by_condition.csv"))

    cond_order = ["C3L", "N", "C3R"]
    cond_label = [COND_LABELS[c] for c in cond_order]

    mat = np.full((len(MODEL_ORDER), len(cond_order)), np.nan)
    for i, model in enumerate(MODEL_ORDER):
        for j, cond in enumerate(cond_order):
            sub = align[(align["model_label"] == model) & (align["condition"] == cond)]
            if not sub.empty:
                mat[i, j] = (sub["closer_to"] == "Dem").mean() * 100

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    im = ax.imshow(mat, cmap="RdBu", vmin=0, vmax=100, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val - 50) > 28 else "#222"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    # Subtle grid between cells
    ax.set_xticks(np.arange(len(cond_order) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(MODEL_ORDER) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(np.arange(len(cond_order)))
    ax.set_xticklabels(cond_label, fontsize=8.5)
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_xlabel("Asker-identity condition", labelpad=8, fontsize=9.5)
    ax.set_title("The conservative-Republican cue flips behavior; the progressive-Democrat cue barely moves it",
                 pad=14, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("% closer to Democrats than Republicans", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()

    save(fig, "fig5_perception_mechanism")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 6 — item-level shift scatter
# ------------------------------------------------------------------
def fig6_itemshift():
    print("\nFigure 6: item-level shift scatter...")

    align = pd.read_csv(os.path.join(DATA_DIR, "phase2_alignment_by_condition.csv"))

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0), sharex=True, sharey=True)
    RED = "#b3412c"
    BLUE = "#2e5c8a"
    for ax, model in zip(axes.flat, MODEL_ORDER):
        sub_n = align[(align["model_label"] == model) & (align["condition"] == "N")].set_index("question_id")
        sub_r = align[(align["model_label"] == model) & (align["condition"] == "C3R")].set_index("question_id")
        sub_l = align[(align["model_label"] == model) & (align["condition"] == "C3L")].set_index("question_id")

        common = sub_n.index.intersection(sub_r.index).intersection(sub_l.index)
        n_diff = (sub_n.loc[common, "wd_rep"] - sub_n.loc[common, "wd_dem"]).values
        r_diff = (sub_r.loc[common, "wd_rep"] - sub_r.loc[common, "wd_dem"]).values
        l_diff = (sub_l.loc[common, "wd_rep"] - sub_l.loc[common, "wd_dem"]).values

        lim = max(abs(n_diff).max(), abs(r_diff).max(), abs(l_diff).max()) * 1.05

        # Shaded "rightward shift" region (below diagonal) — very faint
        ax.fill_between([-lim, lim], [-lim, lim], [-lim, -lim],
                        color=RED, alpha=0.04, zorder=0)
        ax.fill_between([-lim, lim], [-lim, lim], [lim, lim],
                        color=BLUE, alpha=0.04, zorder=0)

        ax.axhline(0, color="#888", linewidth=0.5, alpha=0.6, zorder=1)
        ax.axvline(0, color="#888", linewidth=0.5, alpha=0.6, zorder=1)
        ax.plot([-lim, lim], [-lim, lim], "--", color="#333",
                linewidth=0.8, alpha=0.7, zorder=2)

        ax.scatter(n_diff, l_diff, s=14, alpha=0.45, color=BLUE,
                   edgecolor="none", zorder=3, label="C3L")
        ax.scatter(n_diff, r_diff, s=14, alpha=0.45, color=RED,
                   edgecolor="none", zorder=4, label="C3R")

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"{model}  (n={len(common)})", fontsize=10,
                     fontweight="bold")
        ax.grid(alpha=0.15, linestyle=":", zorder=0)

    for ax in axes[-1]:
        ax.set_xlabel("Baseline  WD(Rep) − WD(Dem)\n(more Rep-like · more Dem-like)",
                      fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Under condition\nWD(Rep) − WD(Dem)", fontsize=9)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="None", color=RED,
                   markersize=8, alpha=0.8, label="C3R (conservative Republican)"),
        plt.Line2D([], [], marker="o", linestyle="None", color=BLUE,
                   markersize=8, alpha=0.8, label="C3L (progressive Democrat)"),
        plt.Line2D([], [], linestyle="--", color="#333",
                   linewidth=1.2, label="No shift (y = x)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.00),
               ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("Item-level distributional shift under partisan identity cues",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()

    save(fig, "fig6_item_framing_distribution")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 3 — open-ended asker perception
# ------------------------------------------------------------------
def fig3_perception():
    print("\nFigure 3: open-ended asker perception...")

    probe_path = os.path.join(DATA_DIR, "results_perception_probe.csv")
    if not os.path.exists(probe_path):
        print("  (skipping: no probe data)")
        return
    df = pd.read_csv(probe_path)
    df = df[df["asker_desc"].notna()]

    # Keyword patterns
    pat_res = r"\b(research|academic|scholar|student|scientist|university|professor|poll|survey|pollster|analyst)"
    pat_partisan = r"\b(progressive|conservative|liberal|democrat|republican|activist)"
    df["is_researcher"] = df["asker_desc"].str.contains(
        pat_res, regex=True, case=False, na=False)
    df["is_partisan"] = df["asker_desc"].str.contains(
        pat_partisan, regex=True, case=False, na=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # Panel A: per-model share describing as researcher/pollster
    y = np.arange(len(MODEL_ORDER))
    vals = []
    for m in MODEL_ORDER:
        sub = df[df["model_label"] == m]
        vals.append(sub["is_researcher"].mean() * 100 if len(sub) else 0)
    ax1.barh(y, vals, color="#2e5c8a", alpha=0.9, edgecolor="white",
             linewidth=0.6, height=0.7, zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels(MODEL_ORDER)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 108)
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.set_xlabel("% responses identifying asker as\nresearcher, pollster, or academic")
    ax1.set_title("Identification as researcher / pollster", pad=10)
    ax1.grid(axis="x", alpha=0.2, linestyle=":", zorder=1)
    for yi, v in zip(y, vals):
        if v >= 20:
            ax1.text(v - 2, yi, f"{v:.0f}%", ha="right", va="center",
                     fontsize=8.5, color="white", fontweight="bold")
        else:
            ax1.text(v + 2, yi, f"{v:.0f}%", ha="left", va="center",
                     fontsize=8.5, color="#222", fontweight="bold")

    # Panel B: political-leaning distribution with shaded bands
    ax2.axvspan(1, 4, facecolor="#2e5c8a", alpha=0.05, zorder=0)
    ax2.axvspan(4, 7, facecolor="#b3412c", alpha=0.05, zorder=0)

    rng = np.random.default_rng(42)
    for i, m in enumerate(MODEL_ORDER):
        sub = df[(df["model_label"] == m) & df["political_leaning"].notna()]
        if sub.empty:
            continue
        vals_p = sub["political_leaning"].values
        jitter_x = rng.uniform(-0.12, 0.12, len(vals_p))
        jitter_y = rng.uniform(-0.18, 0.18, len(vals_p))
        ax2.scatter(vals_p + jitter_x, np.full(len(vals_p), i) + jitter_y,
                    s=12, alpha=0.25, color=MODEL_COLORS[m],
                    edgecolor="none", zorder=2)
        # mean diamond with white halo
        ax2.scatter(vals_p.mean(), i, s=170, color=MODEL_COLORS[m],
                    edgecolor="white", linewidth=1.8, marker="D", zorder=4)
        ax2.scatter(vals_p.mean(), i, s=60, color=MODEL_COLORS[m],
                    edgecolor="black", linewidth=0.5, marker="D", zorder=5)

    ax2.axvline(4, color="black", linewidth=0.9, linestyle="--",
                alpha=0.7, ymin=0.04, ymax=0.96)
    ax2.set_xlim(1, 7)
    ax2.set_ylim(-0.6, len(MODEL_ORDER) - 0.4)
    ax2.set_yticks(range(len(MODEL_ORDER)))
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xticks(range(1, 8))
    ax2.set_xlabel("Inferred political leaning of asker\n(1 = strongly left · 4 = neutral · 7 = strongly right)")
    ax2.set_title("Inferred political leaning", pad=10)
    ax2.grid(axis="x", alpha=0.2, linestyle=":")
    ax2.text(1.1, -0.45, "left", fontsize=7.5, color="#2e5c8a",
             fontweight="bold", ha="left")
    ax2.text(6.9, -0.45, "right", fontsize=7.5, color="#b3412c",
             fontweight="bold", ha="right")
    ax2.text(4, len(MODEL_ORDER) - 0.55, "neutral", fontsize=7.5,
             color="#444", ha="center", style="italic")

    fig.suptitle("Models read the default audit prompt as coming from a neutral academic researcher",
                 fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_asker_perception")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 7 — expected-answer probe
# ------------------------------------------------------------------
def fig7_expected_answer():
    print("\nFigure 7: expected-answer probe...")

    overall = pd.read_csv(os.path.join(DATA_DIR, "expected_answer_overall.csv"))
    per_mod = pd.read_csv(os.path.join(DATA_DIR,
                                       "expected_answer_by_condition_model.csv"))

    cond_order = ["C3L", "N", "C3R"]
    cond_label = {"C3L": "Progressive\nDemocrat (C3L)",
                  "N":   "Default (N)",
                  "C3R": "Conservative\nRepublican (C3R)"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # --- Left: grouped bars, mean asker-wants-Dem vs -Rep across conditions ---
    x = np.arange(len(cond_order))
    bar_w = 0.36
    dem_vals = [overall.set_index("condition").loc[c, "exp_dem"] * 100
                for c in cond_order]
    rep_vals = [overall.set_index("condition").loc[c, "exp_rep"] * 100
                for c in cond_order]

    b1 = ax1.bar(x - bar_w/2, dem_vals, bar_w, color="#2e5c8a",
                 edgecolor="white", linewidth=0.6,
                 label="Asker wants Dem-preferred letter")
    b2 = ax1.bar(x + bar_w/2, rep_vals, bar_w, color="#b3412c",
                 edgecolor="white", linewidth=0.6,
                 label="Asker wants Rep-preferred letter")

    # Value labels placed above the bar top with a small vertical offset
    for bars, vals in [(b1, dem_vals), (b2, rep_vals)]:
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, v + 2,
                     f"{v:.0f}%", ha="center", va="bottom",
                     fontsize=8.5, fontweight="bold", color="#222")

    ax1.set_xticks(x)
    ax1.set_xticklabels([cond_label[c] for c in cond_order], fontsize=8.5)
    ax1.set_ylabel("Share of responses (%)")
    ax1.set_ylim(0, 108)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_title("What letter does the model think the asker wants?", pad=8)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1,
               frameon=False, fontsize=8)
    ax1.grid(axis="y", alpha=0.2, linestyle=":", zorder=0)

    # --- Right: per-model trajectory across conditions ---
    for m in MODEL_ORDER:
        ys = []
        for c in cond_order:
            row = per_mod[(per_mod["model_label"] == m) &
                          (per_mod["condition"] == c)]
            ys.append(row["exp_dem"].iloc[0] * 100 if not row.empty else np.nan)
        ax2.plot(x, ys, "-o", color=MODEL_COLORS[m], linewidth=1.6,
                 markersize=6, markeredgecolor="white", markeredgewidth=0.8,
                 label=m)

    ax2.set_xticks(x)
    ax2.set_xticklabels([cond_label[c] for c in cond_order], fontsize=8.5)
    ax2.set_ylabel("Asker wants Dem-preferred letter (%)")
    ax2.set_ylim(0, 108)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_title("Per-model trajectory across conditions", pad=8)
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
               frameon=False, fontsize=7.5)
    ax2.grid(alpha=0.2, linestyle=":", zorder=0)

    fig.tight_layout()
    save(fig, "fig7_expected_answer")
    plt.close(fig)


if __name__ == "__main__":
    fig1_main()
    fig2_decomp()
    fig3_perception()
    fig5_heatmap()
    fig6_itemshift()
    fig7_expected_answer()
    print("\nAll figures regenerated.")
