"""
analyze_replicates.py — Replicate-stability and greedy-decoding robustness.

Inputs:
  - data/results_phase2.csv               (main experiment; 1 rep at T=1.0)
  - data/results_phase2_replicates.csv    (3 additional reps at T=1.0)
  - data/results_phase2_t0.csv            (1 rep at T=0.0 greedy)
  - data/pew_atp_partisan_distributions.csv  (Dem/Rep empirical distributions)

Outputs (to data/):
  - replicate_stability_summary.txt
  - replicate_per_model_shift.csv
"""

import os, ast, numpy as np, pandas as pd
from scipy.stats import wasserstein_distance

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

LETTERS = "ABCDEFG"
FOCAL = ["N", "C3L", "C3R"]


def load_distributions():
    dist = pd.read_csv(os.path.join(DATA, "pew_atp_partisan_distributions.csv"))
    dist["qid"] = "atp_" + dist["wave"].astype(str) + "_" + dist["var_name"].astype(str)
    def parse_list(s):
        try: return ast.literal_eval(s)
        except: return None
    dist["dem"] = dist["dist_dem"].apply(parse_list)
    dist["rep"] = dist["dist_rep"].apply(parse_list)
    return dist.set_index("qid")[["dem", "rep", "n_options"]].to_dict("index")


def wd_norm(counts, ref):
    n = min(len(counts), len(ref))
    a = np.array(counts[:n], float); b = np.array(ref[:n], float)
    if a.sum() == 0 or b.sum() == 0:
        return np.nan
    a /= a.sum(); b /= b.sum()
    xs = np.arange(n)
    return wasserstein_distance(xs, xs, a, b) / max(1, n - 1)


def cell_wd(df, dists):
    """
    For each (model, qid, cond) cell, aggregate all response letters (across
    available reps) into a single empirical distribution and compute WD(Dem)
    and WD(Rep). Returns tidy dataframe.
    """
    df = df[df["source"] == "pew_atp"]
    df = df[df["condition"].isin(FOCAL)]
    rows = []
    for (m, q, c), g in df.groupby(["model_label", "question_id", "condition"]):
        d = dists.get(q)
        if d is None or d["dem"] is None or d["rep"] is None:
            continue
        n = min(int(d["n_options"]), len(d["dem"]), len(d["rep"]))
        counts = np.zeros(n)
        for L in g["letter"].dropna():
            if isinstance(L, str) and L in LETTERS:
                idx = LETTERS.index(L)
                if idx < n:
                    counts[idx] += 1
        if counts.sum() == 0:
            continue
        rows.append((m, q, c, len(g), int(counts.sum()),
                     wd_norm(counts, d["dem"]), wd_norm(counts, d["rep"])))
    return pd.DataFrame(rows, columns=["model", "qid", "cond",
                                        "n_attempts", "n_valid",
                                        "wd_dem", "wd_rep"])


def summarize(tag, cells):
    """Per-model x cond mean WD(Dem), plus C3R-N and C3L-N shifts."""
    pv = cells.pivot_table(index="model", columns="cond", values="wd_dem")
    out = pv.copy()
    out["C3R-N"] = pv["C3R"] - pv["N"]
    out["C3L-N"] = pv["C3L"] - pv["N"]
    out["asym"] = out["C3R-N"] - out["C3L-N"]
    return out.round(4)


def cell_sd_across_reps(df_all_reps, dists):
    """
    For each (model, qid, cond) cell with >=2 reps, compute WD(Dem) per rep
    (using the single draw per rep), then compute SD across reps. Gives a
    noise estimate at the cell level.
    """
    df = df_all_reps[df_all_reps["source"] == "pew_atp"]
    df = df[df["condition"].isin(FOCAL)]
    per_rep = []
    for (m, q, c, r), g in df.groupby(["model_label", "question_id",
                                         "condition", "rep"]):
        d = dists.get(q)
        if d is None or d["dem"] is None: continue
        n = min(int(d["n_options"]), len(d["dem"]))
        counts = np.zeros(n)
        for L in g["letter"].dropna():
            if isinstance(L, str) and L in LETTERS:
                idx = LETTERS.index(L)
                if idx < n:
                    counts[idx] += 1
        if counts.sum() == 0: continue
        per_rep.append((m, q, c, r, wd_norm(counts, d["dem"])))
    pr = pd.DataFrame(per_rep, columns=["model", "qid", "cond", "rep",
                                         "wd_dem"])
    # Cell-level SD across reps (keep cells with >=2 reps)
    sd = pr.groupby(["model", "qid", "cond"])["wd_dem"].agg(["mean", "std",
                                                               "count"])
    return pr, sd.reset_index()


def main():
    dists = load_distributions()

    print("Loading datasets...")
    main_df = pd.read_csv(os.path.join(DATA, "results_phase2.csv"))
    main_df["rep"] = 0  # original run is rep 0

    reps_df = pd.read_csv(os.path.join(DATA, "results_phase2_replicates.csv"))
    reps_df["rep"] = reps_df["rep"].astype(int) + 1  # reps 1,2,3

    t0_df = pd.read_csv(os.path.join(DATA, "results_phase2_t0.csv"))

    # --- Part 1: T=1.0 replicate stability ---
    print("\n=== T=1.0 replicate stability (4 reps combined vs original 1 rep) ===")
    t1_combined = pd.concat([main_df, reps_df], ignore_index=True,
                            sort=False)
    cells_orig = cell_wd(main_df, dists)
    cells_all4 = cell_wd(t1_combined, dists)
    orig_summary = summarize("orig", cells_orig)
    all4_summary = summarize("4reps", cells_all4)
    print("\nOriginal (1 rep) per-model WD(Dem) and shifts:")
    print(orig_summary)
    print("\nFour reps combined per-model WD(Dem) and shifts:")
    print(all4_summary)

    # Asymmetry stability
    asym_orig = orig_summary["asym"]
    asym_4 = all4_summary["asym"]
    print("\nAsymmetry per model (C3R-N) - (C3L-N):")
    print(pd.DataFrame({"orig_1rep": asym_orig, "4reps": asym_4,
                        "delta": (asym_4 - asym_orig).round(4)}).round(4))

    # Per-cell SD across the 4 reps
    print("\n=== Cell-level SD across the 4 T=1.0 reps ===")
    pr, sd_df = cell_sd_across_reps(t1_combined, dists)
    sd_df = sd_df[sd_df["count"] >= 2]
    print(f"N cells with >=2 reps: {len(sd_df)}")
    print(f"Mean cell SD (WD Dem) across 4 reps: {sd_df['std'].mean():.4f}")
    print(f"Median cell SD: {sd_df['std'].median():.4f}")
    print(f"95th pct cell SD: {sd_df['std'].quantile(0.95):.4f}")
    print("By condition:")
    print(sd_df.groupby("cond")["std"].agg(["mean", "median", "count"])
          .round(4))

    # --- Part 2: T=0 greedy robustness ---
    print("\n=== T=0 greedy decoding vs T=1.0 (original) ===")
    cells_t0 = cell_wd(t0_df, dists)
    t0_summary = summarize("t0", cells_t0)
    print("\nT=0 per-model WD(Dem) and shifts:")
    print(t0_summary)

    print("\nAsymmetry (C3R-N) - (C3L-N) by decoding setting:")
    comp = pd.DataFrame({
        "T=1.0 (4 reps)": all4_summary["asym"],
        "T=0 (greedy)":   t0_summary["asym"],
    }).round(4)
    print(comp)

    # --- Part 3: Write summary ---
    out_txt = os.path.join(DATA, "replicate_stability_summary.txt")
    with open(out_txt, "w") as f:
        f.write("REPLICATE STABILITY AND GREEDY-DECODING ROBUSTNESS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Inputs: original Phase 2 (1 rep, T=1.0) + 3 additional reps "
                "(T=1.0) + 1 rep (T=0.0 greedy).\n")
        f.write(f"Focal grid: 500 ATP items x 6 models x {{N, C3L, C3R}}.\n\n")

        f.write("--- Per-model WD(Dem) and shifts: original vs 4 reps ---\n\n")
        f.write("Original (1 rep):\n" + orig_summary.to_string() + "\n\n")
        f.write("4 reps combined:\n" + all4_summary.to_string() + "\n\n")

        f.write("--- Cell-level SD across 4 T=1.0 reps ---\n")
        f.write(f"N cells (model x item x cond, >=2 reps): {len(sd_df)}\n")
        f.write(f"Mean cell SD in WD(Dem): {sd_df['std'].mean():.4f}\n")
        f.write(f"Median: {sd_df['std'].median():.4f}, "
                f"95th pct: {sd_df['std'].quantile(0.95):.4f}\n")
        f.write("By condition:\n" +
                sd_df.groupby("cond")["std"]
                .agg(["mean", "median", "count"]).round(4).to_string() + "\n\n")

        f.write("--- Asymmetry (C3R-N)-(C3L-N) across settings ---\n")
        comp_full = pd.DataFrame({
            "orig_1rep_T1": asym_orig,
            "4reps_T1":     asym_4,
            "T0_greedy":    t0_summary["asym"],
        }).round(4)
        f.write(comp_full.to_string() + "\n\n")

        f.write("--- Mean shifts across models ---\n")
        for tag, smry in [("orig 1 rep T=1", orig_summary),
                          ("4 reps T=1", all4_summary),
                          ("T=0 greedy", t0_summary)]:
            f.write(f"{tag}: mean C3R-N = {smry['C3R-N'].mean():+.4f}, "
                    f"mean C3L-N = {smry['C3L-N'].mean():+.4f}, "
                    f"mean asym = {smry['asym'].mean():+.4f}\n")

    print(f"\nWrote {out_txt}")

    # Per-model per-cond shift CSV
    csv_path = os.path.join(DATA, "replicate_per_model_shift.csv")
    all4_summary.to_csv(csv_path)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
