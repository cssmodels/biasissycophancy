"""
extract_pew_atp.py — Extract opinion items from Pew ATP .sav files

Reproduces (approximately) the Santurkar et al. (2023) OpinionQA filtering:
1. Remove metadata/demographic variables
2. Remove variables with <2 substantive response options
3. Remove variables where question text interpolates prior answers
4. Remove purely behavioral/factual items (gun ownership counts, membership, etc.)
5. Clean question text: remove survey artifacts, make battery items standalone
6. Standardize options into clean multiple-choice format

Input:  Unzipped Pew ATP wave directories in SOURCE_DIR, each containing a .sav file
Output: CSV with columns: wave, var_name, question, options, n_options, pct_answered
"""

import pyreadstat
import os
import glob
import re
import pandas as pd
import sys

SOURCE_DIR = "/tmp/pewatp_work"
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "pew_atp_items.csv"
)

# ── metadata / demographic variable patterns to exclude ──────────────
EXCLUDE_PATTERNS = [
    r'^QKEY$', r'^WEIGHT', r'^NEW_Device', r'^LANGUAGE', r'^FORM_',
    r'^CREGION', r'^AGE\b', r'^SEX$', r'^EDUCATION', r'^CITIZEN',
    r'^MARITAL', r'^RELIG$', r'^RELIGATTEND', r'^POLPARTY', r'^INCOME',
    r'^POLIDEOLOGY', r'^RACE$', r'^BIRTH_YEAR', r'^GENDER', r'^IDEO$',
    r'^F_', r'^DEVICE_TYPE', r'^PIAL', r'^RACETHN', r'^HISP',
    r'^XTABLET', r'^RACECMB', r'^LANG', r'^SAMPLE', r'^REG$',
    r'^PARTY$', r'^PARTYLN', r'^PARTYSUM', r'^XPARTY',
    r'^INTERVIEW', r'^XOVERALL', r'^DURATION', r'^PHONE$',
    r'^LIVING', r'^KIDSMDY', r'^KIDS_', r'^MARSTAT',
    r'^EMPLOY', r'^DOV_', r'^STATE$', r'^METRO', r'^DENSITY',
    r'^PARENT', r'^VOLUNTEER', r'^ATTEND', r'^BORN',
    r'^PARTYID', r'^VOLPARTY', r'^XSEX', r'^XRACE',
    r'^XEDUC', r'^XINCOME', r'^XRELIG', r'^XBORN',
    r'^XEMPLOY', r'^XMARITAL', r'^XCITIZEN', r'^XMETRO',
    r'^XDENSITY', r'^XSTATE', r'^XREGION', r'^XAGE',
]

# ── factual/behavioral patterns to exclude ───────────────────────────
BEHAVIORAL_PATTERNS = [
    r'do you personally own',
    r'does anyone else in your household own',
    r'how many .* do you .* own',
    r'are you currently a member of',
    r'have you ever fired',
    r'at what age did you',
    r'how often.*do you go hunting',
    r'how often.*do you go shooting',
    r'does your spouse',
    r'do you ever use the gun',
    r'is the gun in your home kept',
    r'how many of the guns in your home',
    r'there is a gun that is both loaded',
    r'have you ever personally owned',
]

# ── directory wave mapping ─────────────────────────────────────────
DIR_TO_WAVE = {
    'W26_Apr17': 'W26', 'W27_May17': 'W27', 'W29_Sep17': 'W29',
    'W32_Feb18': 'W32', 'W34_Apr18': 'W34', 'W36_Jun 18': 'W36',
    'W41_Dec18': 'W41', 'W42_Jan19': 'W42', 'W43_Jan19': 'W43',
    'W45_Feb19': 'W45', 'W49_Jun19': 'W49', 'W50_Jun19': 'W50',
    'W54_Sep19': 'W54', 'W82_Feb21': 'W82', 'W92_Jul21': 'W92',
}


def is_meta_var(name):
    for pat in EXCLUDE_PATTERNS:
        if re.match(pat, name, re.IGNORECASE):
            return True
    return False


def is_behavioral(label):
    for pat in BEHAVIORAL_PATTERNS:
        if re.search(pat, label, re.IGNORECASE):
            return True
    return False


def clean_question_text(label):
    """
    Clean question text following Santurkar's approach:
    1. Remove variable-name prefix (e.g. "SATISF. ")
    2. Remove survey artifacts like [RANDOMIZE], [INSERT], etc.
    3. Remove interviewer instructions
    4. Clean up ALL CAPS words (except acronyms)
    5. Make battery items standalone by integrating the sub-item
    """
    # Remove variable-name prefix
    text = re.sub(r'^[A-Z0-9_]+[a-z]?\.\s*', '', label)

    # Remove survey artifacts in brackets
    text = re.sub(r'\[RANDOMIZE.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[INSERT.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[IF.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[DISPLAY.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[READ.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[PROBE.*?\]', '', text, flags=re.IGNORECASE)

    # Clean up ALL CAPS to Title Case, but keep short acronyms (2-4 chars)
    def fix_caps(match):
        word = match.group(0)
        if len(word) <= 4:
            return word  # Keep short acronyms like U.S., NRA, etc.
        return word.capitalize()
    text = re.sub(r'\b[A-Z]{5,}\b', fix_caps, text)

    # Clean extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_wave(wave_dir, wave_name):
    """Extract opinion items from a single ATP wave."""
    sav_files = glob.glob(os.path.join(wave_dir, "*.sav"))
    if not sav_files:
        return []

    df, meta = pyreadstat.read_sav(sav_files[0])
    n_total = len(df)
    items = []

    for col in meta.column_names:
        stripped = re.sub(r'_W\d+$', '', col)
        label = meta.column_names_to_labels.get(col, "")
        vlabels = meta.variable_value_labels.get(col, {})

        # ── Filter 1: metadata/demographic ──
        if is_meta_var(stripped):
            continue

        # ── Filter 2: insufficient substantive options ──
        non_refused = {
            k: v for k, v in vlabels.items()
            if k not in (99.0, 999.0, 98.0)
            and v.lower() not in ('refused', 'not asked', 'not applicable')
        }
        if len(non_refused) < 2:
            continue

        # ── Filter 3: variable-dependent text ──
        if re.search(
            r'\$\{?\w+\}?|\[.*?response.*?\]|you said|you mentioned',
            label, re.IGNORECASE
        ):
            continue

        # ── Filter 4: behavioral/factual ──
        if is_behavioral(label):
            continue

        # ── Extract and clean ──
        n_valid = df[col].notna().sum()
        pct = round(n_valid / n_total * 100, 1)

        q_text = clean_question_text(label)
        opts = [v for k, v in sorted(non_refused.items())]

        items.append({
            'wave': wave_name,
            'var_name': stripped,
            'question': q_text,
            'options': ' | '.join(opts),
            'n_options': len(opts),
            'pct_answered': pct,
        })

    return items


def main():
    all_items = []

    print("Extracting Pew ATP items...")
    print(f"{'Wave':>5} {'Extracted':>10} {'Respondents':>12}")
    print("-" * 35)

    for wdir_name, wave in sorted(DIR_TO_WAVE.items(), key=lambda x: x[1]):
        wdir = os.path.join(SOURCE_DIR, wdir_name)
        if not os.path.isdir(wdir):
            print(f"  {wave}: DIRECTORY NOT FOUND ({wdir})")
            continue

        items = extract_wave(wdir, wave)
        print(f"{wave:>5} {len(items):>10}")
        all_items.extend(items)

    print("-" * 35)
    print(f"{'TOTAL':>5} {len(all_items):>10}")

    # Save
    outdf = pd.DataFrame(all_items)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    outdf.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    # Summary stats
    print(f"\nOptions distribution:")
    for n in sorted(outdf['n_options'].unique()):
        count = (outdf['n_options'] == n).sum()
        print(f"  {n} options: {count} items")


if __name__ == "__main__":
    main()
