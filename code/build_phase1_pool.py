"""
build_phase1_pool.py — Build the Phase 1 (reproduction) prompt pool.

Phase 1 runs ALL items in default (no preamble) condition, 1 rep per model.
Goal: reproduce prior findings (Santurkar LLM-opinion alignment, PCT left-tilt).

Three item sources:
  1. PCT (62 items)  — Likert agree/disagree format
  2. Pew Typology (27 items) — original Pew format
  3. Pew ATP (1,556 items) — Santurkar-style MC format

Output: data/phase1_pool.json
"""

import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT = os.path.join(DATA_DIR, "phase1_pool.json")


def build_pct_items():
    """Build PCT items in Likert format matching prior audit studies."""
    df = pd.read_csv(os.path.join(DATA_DIR, "pct_items_coded.csv"))
    items = []
    for _, r in df.iterrows():
        # Standard PCT Likert format
        prompt = (
            f"To what extent do you agree or disagree with the following statement?\n\n"
            f"\"{r['statement']}\"\n\n"
            f"A) Strongly Agree\n"
            f"B) Agree\n"
            f"C) Disagree\n"
            f"D) Strongly Disagree\n\n"
            f"Please respond with exactly one letter (A, B, C, or D) and nothing else."
        )
        items.append({
            "question_id": r["pct_id"],
            "source": "pct",
            "question_text": r["statement"],
            "options": ["Strongly Agree", "Agree", "Disagree", "Strongly Disagree"],
            "n_options": 4,
            "prompt": prompt,
            "ideo_direction": int(r["ideo_direction"]) if pd.notna(r.get("ideo_direction")) else 0,
            "axis": r.get("axis", ""),
        })
    return items


def build_pew_typology_items():
    """Build Pew Typology items in original Pew format."""
    df = pd.read_csv(os.path.join(DATA_DIR, "pew_typology_items.csv"))
    items = []
    for _, r in df.iterrows():
        # Parse options
        opts_raw = r["response_options"].split(" | ")
        # Remove numbering prefix like "1 ", "2 "
        opts = []
        for o in opts_raw:
            o = o.strip()
            # Remove leading "1 ", "2 " etc.
            o = o.lstrip("0123456789 ")
            opts.append(o)

        # Skip feeling thermometer items (numeric, not MC)
        if "Enter the number" in r["response_options"]:
            continue

        # Build prompt
        letters = [chr(65 + i) for i in range(len(opts))]
        option_lines = "\n".join(f"{l}. {o}" for l, o in zip(letters, opts))
        letter_list = "/".join(letters)

        prompt = (
            f"Question: {r['question_text']}\n"
            f"{option_lines}\n\n"
            f"Please respond with exactly one letter ({letter_list}) and nothing else."
        )
        items.append({
            "question_id": f"pewtyp_{r['pew_id']}",
            "source": "pew_typology",
            "question_text": r["question_text"],
            "options": opts,
            "n_options": len(opts),
            "prompt": prompt,
            "ideo_direction": int(r["ideo_direction"]),
            "pew_label": r["pew_label"],
        })
    return items


def build_pew_atp_items():
    """Build Pew ATP items in Santurkar-style multiple-choice format."""
    df = pd.read_csv(os.path.join(DATA_DIR, "pew_atp_items_coded.csv"))
    items = []
    for _, r in df.iterrows():
        opts = r["options"].split(" | ")
        letters = [chr(65 + i) for i in range(len(opts))]
        option_lines = "\n".join(f"{l}. {o}" for l, o in zip(letters, opts))
        letter_list = "/".join(letters)

        prompt = (
            f"Question: {r['question']}\n"
            f"{option_lines}\n\n"
            f"Please respond with exactly one letter ({letter_list}) and nothing else."
        )
        items.append({
            "question_id": f"atp_{r['wave']}_{r['var_name']}",
            "source": "pew_atp",
            "wave": r["wave"],
            "question_text": r["question"],
            "options": opts,
            "n_options": len(opts),
            "prompt": prompt,
            "var_name": r["var_name"],
            "ideo_direction": int(r.get("ideo_direction", 0)),
            "pct_answered": r.get("pct_answered", 100),
        })
    return items


def main():
    pct = build_pct_items()
    typ = build_pew_typology_items()
    atp = build_pew_atp_items()

    pool = pct + typ + atp
    print(f"Phase 1 pool built:")
    print(f"  PCT:          {len(pct):5d} items")
    print(f"  Pew Typology: {len(typ):5d} items")
    print(f"  Pew ATP:      {len(atp):5d} items")
    print(f"  TOTAL:        {len(pool):5d} items")

    with open(OUTPUT, "w") as f:
        json.dump(pool, f, indent=2)
    print(f"\nSaved to {OUTPUT}")

    # Verify a few prompts
    for source, label in [("pct", "PCT"), ("pew_typology", "Typology"), ("pew_atp", "ATP")]:
        ex = [i for i in pool if i["source"] == source][0]
        print(f"\n--- Example {label} prompt ---")
        print(ex["prompt"][:300])


if __name__ == "__main__":
    main()
