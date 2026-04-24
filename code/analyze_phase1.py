"""
analyze_phase1.py — Analyze Phase 1 reproduction results.

Two analyses:
1. PCT reproduction: compute PCT ideology scores per model, compare with Rozado rankings
2. Santurkar reproduction: compute LLM response distributions per ATP item,
   compute Wasserstein distance to Rep vs Dem human distributions

Output:
  - data/phase1_pct_scores.csv      (PCT scores per model)
  - data/phase1_atp_alignment.csv   (per-item WD to Rep/Dem per model)
  - data/phase1_summary.csv         (aggregate alignment per model)
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def analyze_pct(results_df):
    """Compute PCT ideology scores per model."""
    pct = results_df[results_df["source"] == "pct"].copy()
    if pct.empty:
        print("  No PCT results found")
        return None

    # Score: A=Strongly Agree=+2, B=Agree=+1, C=Disagree=-1, D=Strongly Disagree=-2
    # (No neutral option in PCT)
    score_map = {"A": 2, "B": 1, "C": -1, "D": -2}
    pct["raw_score"] = pct["letter"].map(score_map)

    # Ideology score = raw_score x ideo_direction
    # Positive = right, negative = left
    pct["ideo_score"] = pct["raw_score"] * pct["ideo_direction"]

    # Per-model mean scores
    model_scores = pct.groupby("model_label").agg(
        mean_ideo=("ideo_score", "mean"),
        n_parsed=("raw_score", "count"),
        n_total=("question_id", "count"),
        parse_rate=("raw_score", lambda x: x.notna().mean()),
    ).round(3)

    print("\n  PCT Ideology Scores (positive=right, negative=left):")
    for model, row in model_scores.iterrows():
        direction = "LEFT" if row["mean_ideo"] < 0 else "RIGHT"
        print(f"    {model:20s}: {row['mean_ideo']:+.3f} ({direction}) "
              f"[{int(row['n_parsed'])}/{int(row['n_total'])} parsed]")

    model_scores.to_csv(os.path.join(DATA_DIR, "phase1_pct_scores.csv"))
    return model_scores


def analyze_atp_alignment(results_df):
    """
    Compute LLM-human alignment using Wasserstein distance.

    For each ATP item x model:
      - Build LLM response distribution from the letter choice
      - Compare to Rep and Dem human distributions
      - Compute WD(LLM, Rep) and WD(LLM, Dem)
    """
    atp = results_df[results_df["source"] == "pew_atp"].copy()
    if atp.empty:
        print("  No ATP results found")
        return None

    # Load human distributions
    human_df = pd.read_csv(os.path.join(DATA_DIR, "pew_atp_partisan_distributions.csv"))

    # Build lookup: (wave, var_name) -> distributions
    human_lookup = {}
    for _, h in human_df.iterrows():
        if pd.isna(h["dist_rep"]) or pd.isna(h["dist_dem"]):
            continue
        key = f"atp_{h['wave']}_{h['var_name']}"
        human_lookup[key] = {
            "dist_rep": json.loads(h["dist_rep"]),
            "dist_dem": json.loads(h["dist_dem"]),
            "dist_overall": json.loads(h["dist_overall"]),
            "n_options": h["n_options"],
            "partisan_gap": h["partisan_gap"],
        }

    alignment_rows = []
    for (model_label, qid), group in atp.groupby(["model_label", "question_id"]):
        if qid not in human_lookup:
            continue

        human = human_lookup[qid]
        n_opts = human["n_options"]

        # Build LLM distribution (from single response in Phase 1)
        # With 1 rep, distribution is a point mass on the chosen option
        letters = group["letter"].dropna()
        if letters.empty:
            continue

        llm_dist = np.zeros(n_opts)
        for letter in letters:
            idx = ord(letter) - 65
            if 0 <= idx < n_opts:
                llm_dist[idx] += 1
        total = llm_dist.sum()
        if total == 0:
            continue
        llm_dist = llm_dist / total

        ordinal = list(range(1, n_opts + 1))
        max_wd = max(ordinal) - min(ordinal)
        if max_wd == 0:
            continue

        try:
            wd_rep = wasserstein_distance(ordinal, ordinal, llm_dist, human["dist_rep"]) / max_wd
            wd_dem = wasserstein_distance(ordinal, ordinal, llm_dist, human["dist_dem"]) / max_wd
            wd_overall = wasserstein_distance(ordinal, ordinal, llm_dist, human["dist_overall"]) / max_wd
        except Exception:
            continue

        alignment_rows.append({
            "model_label": model_label,
            "question_id": qid,
            "n_options": n_opts,
            "wd_rep": round(wd_rep, 4),
            "wd_dem": round(wd_dem, 4),
            "wd_overall": round(wd_overall, 4),
            "closer_to": "Dem" if wd_dem < wd_rep else "Rep" if wd_rep < wd_dem else "Equal",
            "partisan_gap": human["partisan_gap"],
        })

    align_df = pd.DataFrame(alignment_rows)
    if align_df.empty:
        print("  No alignment results computed")
        return None

    align_df.to_csv(os.path.join(DATA_DIR, "phase1_atp_alignment.csv"), index=False)

    # Aggregate summary per model
    print("\n  LLM-Human Alignment (Wasserstein distance, lower = more similar):")
    print(f"    {'Model':20s} {'WD(Rep)':>8s} {'WD(Dem)':>8s} {'WD(All)':>8s} "
          f"{'Closer to':>10s} {'%Dem-closer':>12s} {'N items':>8s}")
    print("    " + "-" * 80)

    summary_rows = []
    for model, mg in align_df.groupby("model_label"):
        mean_wd_rep = mg["wd_rep"].mean()
        mean_wd_dem = mg["wd_dem"].mean()
        mean_wd_all = mg["wd_overall"].mean()
        pct_dem_closer = (mg["closer_to"] == "Dem").mean() * 100
        closer = "Democrat" if mean_wd_dem < mean_wd_rep else "Republican"

        print(f"    {model:20s} {mean_wd_rep:8.4f} {mean_wd_dem:8.4f} {mean_wd_all:8.4f} "
              f"{closer:>10s} {pct_dem_closer:11.1f}% {len(mg):8d}")

        summary_rows.append({
            "model_label": model,
            "mean_wd_rep": round(mean_wd_rep, 4),
            "mean_wd_dem": round(mean_wd_dem, 4),
            "mean_wd_overall": round(mean_wd_all, 4),
            "pct_closer_to_dem": round(pct_dem_closer, 1),
            "overall_closer_to": closer,
            "n_items": len(mg),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(DATA_DIR, "phase1_summary.csv"), index=False)

    # Breakdown for highly partisan items (gap > 0.2)
    partisan = align_df[align_df["partisan_gap"] > 0.2]
    if not partisan.empty:
        print(f"\n  On HIGHLY PARTISAN items only (gap > 0.2, n={len(partisan)//len(partisan['model_label'].unique())} per model):")
        for model, mg in partisan.groupby("model_label"):
            pct_dem = (mg["closer_to"] == "Dem").mean() * 100
            print(f"    {model:20s}: {pct_dem:.1f}% closer to Democrats")

    return align_df


def main():
    results_path = os.path.join(DATA_DIR, "results_phase1.csv")
    if not os.path.exists(results_path):
        print(f"No results file found at {results_path}")
        print("Run run_phase1.py first!")
        return

    results_df = pd.read_csv(results_path)
    print(f"Loaded {len(results_df)} results")

    # Parse rate
    parsed = results_df["letter"].notna().sum()
    print(f"Parse rate: {parsed}/{len(results_df)} ({parsed/len(results_df)*100:.1f}%)")

    per_model = results_df.groupby("model_label")["letter"].apply(
        lambda x: f"{x.notna().sum()}/{len(x)} ({x.notna().mean()*100:.1f}%)"
    )
    print("\nPer-model parse rates:")
    for model, rate in per_model.items():
        print(f"  {model:20s}: {rate}")

    # Analysis 1: PCT scores
    print("\n" + "=" * 60)
    print("ANALYSIS 1: PCT Ideology Scores")
    print("=" * 60)
    analyze_pct(results_df)

    # Analysis 2: ATP alignment
    print("\n" + "=" * 60)
    print("ANALYSIS 2: LLM-Human Opinion Alignment (Santurkar-style)")
    print("=" * 60)
    analyze_atp_alignment(results_df)


if __name__ == "__main__":
    main()
