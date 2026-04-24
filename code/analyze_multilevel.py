"""
analyze_multilevel.py — SI robustness check: multilevel model + permutation test.

Replaces reliance on cross-item Wilcoxon p-values and the cross-model
paired t-test with:
  (a) a linear mixed-effects model on per-item Wasserstein distance to
      Democrats, with random intercepts for item and for model and a
      fixed effect for condition (N, C3L, C3R), restricted to the ATP
      500-item Phase-2 partisan pool.
  (b) a within-model, within-item permutation test of the C3R-vs-C3L
      asymmetry, permuting condition labels within each model-item pair.

Outputs: prints a short summary; writes data/multilevel_summary.txt and
data/permutation_summary.txt.
"""

import os, sys, numpy as np, pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

df = pd.read_csv(os.path.join(DATA_DIR, "phase2_alignment_by_condition.csv"))
df = df[df["question_id"].str.startswith("atp_")].copy()
df = df[df["condition"].isin(["N","C3L","C3R"])].copy()

# Outcome: wd_dem (smaller = closer to Democrats).
# Build design for MLM.
try:
    import statsmodels.formula.api as smf
    # Use model and question_id as random effects.
    # statsmodels only supports one grouping; we use question_id crossed via
    # variance components.
    df["cond"] = pd.Categorical(df["condition"], categories=["N","C3L","C3R"])
    vc = {"model": "0 + C(model_label)"}
    md = smf.mixedlm("wd_dem ~ C(cond, Treatment('N'))",
                     data=df, groups=df["question_id"],
                     vc_formula=vc, re_formula="0")
    mdf = md.fit(method="lbfgs", reml=True)
    with open(os.path.join(DATA_DIR, "multilevel_summary.txt"), "w") as f:
        f.write(mdf.summary().as_text())
    print("=== Multilevel model (random intercepts: item, model) ===")
    print(mdf.summary().tables[1])
    params = mdf.params
    bse = mdf.bse
    for term in ["C(cond, Treatment('N'))[T.C3L]",
                 "C(cond, Treatment('N'))[T.C3R]"]:
        if term in params.index:
            b = params[term]; se = bse[term]
            z = b/se
            ci_lo, ci_hi = b-1.96*se, b+1.96*se
            print(f"  {term}: b = {b:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], z = {z:.1f}")
except Exception as e:
    print("MLM failed:", e)

# Permutation test for the C3R-vs-C3L asymmetry.
# For each (model, question_id) present under both C3L and C3R,
# compute shift_dem(C3R) - shift_dem(C3L). Observed statistic is the
# mean of these. Under the null (condition label uninformative), we can
# permute the (C3L, C3R) labels within each (model, item) pair.
wide = df.pivot_table(index=["model_label","question_id"],
                       columns="condition", values="wd_dem").dropna()
wide["shift_c3l"] = wide["C3L"] - wide["N"]
wide["shift_c3r"] = wide["C3R"] - wide["N"]
wide["asym"] = wide["shift_c3r"] - wide["shift_c3l"]
obs = wide["asym"].mean()

rng = np.random.default_rng(42)
B = 10000
# Under null, the (C3L - N, C3R - N) pair is interchangeable within cell.
# So we flip the sign of "asym" independently for each cell.
signs = rng.choice([-1, 1], size=(B, len(wide)))
perm_means = (signs * wide["asym"].to_numpy()).mean(axis=1)
p_perm = (np.abs(perm_means) >= abs(obs)).mean()

# Also: per-model cluster-preserving permutation (shuffle signs at the model
# level, i.e., flip all cells within a model together).
m_groups = wide.reset_index().groupby("model_label")["asym"].mean()
B2 = 2**len(m_groups)  # 64
obs_model = m_groups.mean()
from itertools import product
perm_model_means = [ (np.array(sv) * m_groups.values).mean()
                     for sv in product([-1, 1], repeat=len(m_groups)) ]
p_model = float(np.mean(np.abs(perm_model_means) >= abs(obs_model)))

with open(os.path.join(DATA_DIR, "permutation_summary.txt"), "w") as f:
    f.write(f"Observed mean (C3R−N) − (C3L−N) in WD(Dem): {obs:+.4f}\n")
    f.write(f"Item-level sign permutation test (B=10000): p = {p_perm:.4f}\n")
    f.write(f"\nPer-model means of asymmetry:\n{m_groups.to_string()}\n")
    f.write(f"\nModel-cluster permutation test (exact, 64 perms): p = {p_model:.4f}\n")

print("\n=== Permutation tests for C3R-vs-C3L asymmetry ===")
print(f"Observed mean asymmetry (C3R−N) − (C3L−N) in WD(Dem): {obs:+.4f}")
print(f"Item-level sign-flip permutation test (B={B}): p = {p_perm:.4f}")
print(f"\nPer-model asymmetry means:")
print(m_groups.round(4).to_string())
print(f"\nModel-cluster exact permutation test (64 perms): p = {p_model:.4f}")
