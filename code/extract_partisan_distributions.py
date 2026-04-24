"""
extract_partisan_distributions.py — Compute per-item response distributions
for Republicans vs Democrats from raw Pew ATP .sav files.

For each survey item, produces:
  - Overall response distribution
  - Rep/lean-Rep response distribution
  - Dem/lean-Dem response distribution
  - Partisan disagreement score (Wasserstein distance between R and D distributions)

Uses F_PARTYSUM_FINAL (1=Rep/lean Rep, 2=Dem/lean Dem) available in all 15 waves.

Output: data/pew_atp_partisan_distributions.csv
"""

import pyreadstat
import numpy as np
import pandas as pd
import os
import glob
import re
import json
from scipy.stats import wasserstein_distance

SOURCE_DIR = "/tmp/pewatp_work"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ITEMS_FILE = os.path.join(DATA_DIR, "pew_atp_items_coded.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "pew_atp_partisan_distributions.csv")

DIR_TO_WAVE = {
    'W26_Apr17': 'W26', 'W27_May17': 'W27', 'W29_Sep17': 'W29',
    'W32_Feb18': 'W32', 'W34_Apr18': 'W34', 'W36_Jun 18': 'W36',
    'W41_Dec18': 'W41', 'W42_Jan19': 'W42', 'W43_Jan19': 'W43',
    'W45_Feb19': 'W45', 'W49_Jun19': 'W49', 'W50_Jun19': 'W50',
    'W54_Sep19': 'W54', 'W82_Feb21': 'W82', 'W92_Jul21': 'W92',
}
WAVE_TO_DIR = {v: k for k, v in DIR_TO_WAVE.items()}


def compute_distribution(series, value_labels):
    """Compute normalized distribution over non-refused options."""
    non_refused_keys = sorted(
        k for k in value_labels
        if k not in (99.0, 999.0, 98.0)
        and value_labels[k].lower() not in ('refused', 'not asked', 'not applicable')
    )
    if not non_refused_keys:
        return None, None

    counts = []
    labels = []
    for k in non_refused_keys:
        n = (series == k).sum()
        counts.append(n)
        labels.append(value_labels[k])

    total = sum(counts)
    if total == 0:
        return None, None

    dist = [c / total for c in counts]
    return dist, labels


def compute_partisan_gap(dist_r, dist_d, n_options):
    """Wasserstein distance between R and D distributions, normalized by max possible."""
    if dist_r is None or dist_d is None:
        return None
    ordinal = list(range(1, n_options + 1))
    try:
        wd = wasserstein_distance(ordinal, ordinal, dist_r, dist_d)
        max_wd = max(ordinal) - min(ordinal)  # max possible WD
        return wd / max_wd if max_wd > 0 else 0
    except Exception:
        return None


def main():
    # Load our item list
    items_df = pd.read_csv(ITEMS_FILE)
    print(f"Processing {len(items_df)} items across {items_df['wave'].nunique()} waves...")

    results = []

    for wave, wave_items in items_df.groupby('wave'):
        wdir_name = WAVE_TO_DIR.get(wave)
        if not wdir_name:
            print(f"  {wave}: directory not found, skipping")
            continue

        wdir = os.path.join(SOURCE_DIR, wdir_name)
        sav_files = glob.glob(os.path.join(wdir, "*.sav"))
        if not sav_files:
            continue

        df, meta = pyreadstat.read_sav(sav_files[0])

        # Partisan ID
        party_col = 'F_PARTYSUM_FINAL'
        if party_col not in df.columns:
            print(f"  {wave}: no {party_col}, skipping")
            continue

        rep_mask = df[party_col] == 1.0  # Rep/lean Rep
        dem_mask = df[party_col] == 2.0  # Dem/lean Dem
        n_rep = rep_mask.sum()
        n_dem = dem_mask.sum()

        processed = 0
        for _, item in wave_items.iterrows():
            var_name = item['var_name']

            # Find the actual column (may have _W## suffix)
            wave_num = wave[1:]
            candidates = [
                f"{var_name}_W{wave_num}",
                var_name,
            ]
            col = None
            for c in candidates:
                if c in df.columns:
                    col = c
                    break
            if col is None:
                # Try case-insensitive
                for c in df.columns:
                    if c.lower() == candidates[0].lower() or c.lower() == var_name.lower():
                        col = c
                        break

            if col is None:
                results.append({
                    'wave': wave,
                    'var_name': var_name,
                    'question': item['question'][:100],
                    'n_options': item['n_options'],
                    'dist_overall': None,
                    'dist_rep': None,
                    'dist_dem': None,
                    'option_labels': None,
                    'n_overall': 0,
                    'n_rep': 0,
                    'n_dem': 0,
                    'partisan_gap': None,
                    'error': 'column_not_found',
                })
                continue

            vlabels = meta.variable_value_labels.get(col, {})

            # Overall distribution
            dist_all, opt_labels = compute_distribution(df[col], vlabels)
            if dist_all is None:
                continue

            # Partisan distributions
            dist_r, _ = compute_distribution(df.loc[rep_mask, col], vlabels)
            dist_d, _ = compute_distribution(df.loc[dem_mask, col], vlabels)

            # Partisan gap
            gap = compute_partisan_gap(dist_r, dist_d, len(opt_labels))

            n_all_valid = df[col].notna().sum()
            n_rep_valid = df.loc[rep_mask, col].notna().sum()
            n_dem_valid = df.loc[dem_mask, col].notna().sum()

            results.append({
                'wave': wave,
                'var_name': var_name,
                'question': item['question'][:200],
                'n_options': len(opt_labels),
                'option_labels': json.dumps(opt_labels),
                'dist_overall': json.dumps([round(x, 4) for x in dist_all]),
                'dist_rep': json.dumps([round(x, 4) for x in dist_r]) if dist_r else None,
                'dist_dem': json.dumps([round(x, 4) for x in dist_d]) if dist_d else None,
                'n_overall': n_all_valid,
                'n_rep': n_rep_valid,
                'n_dem': n_dem_valid,
                'partisan_gap': round(gap, 4) if gap is not None else None,
                'error': None,
            })
            processed += 1

        print(f"  {wave}: {processed}/{len(wave_items)} items processed "
              f"(R={n_rep}, D={n_dem})")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False)

    # Summary stats
    valid = out_df[out_df['partisan_gap'].notna()]
    print(f"\n{'='*50}")
    print(f"Total items with distributions: {len(valid)}")
    print(f"Items not found in .sav: {(out_df['error'] == 'column_not_found').sum()}")
    print(f"\nPartisan gap statistics:")
    print(f"  Mean:   {valid['partisan_gap'].mean():.4f}")
    print(f"  Median: {valid['partisan_gap'].median():.4f}")
    print(f"  Std:    {valid['partisan_gap'].std():.4f}")
    print(f"  Max:    {valid['partisan_gap'].max():.4f}")

    # Top 20 most partisan items
    print(f"\nTop 20 most partisan items:")
    for _, r in valid.nlargest(20, 'partisan_gap').iterrows():
        print(f"  {r['partisan_gap']:.3f} | {r['wave']} {r['var_name']}: "
              f"{r['question'][:70]}")

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
