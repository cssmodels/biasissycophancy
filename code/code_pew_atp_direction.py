"""
code_pew_atp_direction.py — Programmatic ideological direction coding of Pew ATP items.

Heuristic approach:
1. Check response options for known left/right paired framings
2. Check question topic keywords
3. Flag as ambiguous when signal is weak

Coding convention:
  ideo_direction =  1  agreeing / choosing option A is RIGHT-coded
  ideo_direction = -1  agreeing / choosing option A is LEFT-coded
  ideo_direction =  0  ambiguous / non-ideological

These are PROVISIONAL codings for quantitative analysis. They should be
reviewed manually for any items used in the focal factorial experiment.
"""

import pandas as pd
import re
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
INPUT = os.path.join(DATA_DIR, "pew_atp_items.csv")
OUTPUT = os.path.join(DATA_DIR, "pew_atp_items_coded.csv")


# ── LEFT-coded option keywords ───────────────────────────────────────
# If option A/first option matches these, the item is LEFT-coded (ideo_direction = -1)
LEFT_OPTION_PATTERNS = [
    r'government should do more',
    r'bigger government',
    r'too much profit',
    r'unfairly favors powerful',
    r'systemic(ally)? racist',
    r'fundamentally biased',
    r'stricter gun',
    r'ban.*(assault|guns|weapons)',
    r'more difficult.*(buy|obtain|purchase).*(gun|firearm)',
    r'too easy.*(buy|obtain|get).*(gun|firearm)',
    r'too much time in prison',
    r'climate change.*(serious|major|significant)',
    r'human activit(y|ies)',
    r'openness.*(essential|important)',
    r'obstacles.*still.*significant',
    r'a lot more needs to be done',
    r'completely rebuilt',
    r'very good for society',  # re: trans acceptance
    r'legal in all',  # abortion
    r'favor.*(marijuana|cannabis)',
    r'government aid.*more good',
    r'religion.*separate.*government',
    r'accept.*(gay|same-sex) marriage',
    r'too little.*(regulate|regulation)',
    r'corporations.*too much',
    r'helps? a lot$',  # re: being white helps
    r'great deal$',    # re: white advantage
]

# ── RIGHT-coded option keywords ──────────────────────────────────────
RIGHT_OPTION_PATTERNS = [
    r'government is doing too many',
    r'smaller government',
    r'fair and reasonable.*profit',
    r'generally fair to most',
    r'protect.*(gun|second amendment|2nd)',
    r'right to own',
    r'not the government.s job',
    r'stands? above all other',
    r'only military superpower',
    r'too little time in prison',
    r'obstacles.*largely gone',
    r'risk losing.*identity',
    r'nothing at all$',  # re: racial equality nothing needed
    r'illegal in all',  # abortion
    r'oppose.*(marijuana|cannabis)',
    r'government aid.*more harm',
    r'government.*(support|promote).*religious',
    r'not at all$',  # re: white advantage
    r'traditional.*marriage',
    r'too much.*(regulate|regulation)',
    r'corporations.*fair',
]

# ── Topic-based LEFT keywords in question text ───────────────────────
LEFT_TOPIC_PATTERNS = [
    (r'climate change|global warming|carbon|fossil fuel', -1),
    (r'racial (discrimination|inequality|injustice)', -1),
    (r'gender (pay )?gap|gender inequality|sexism', -1),
    (r'immigration.*benefit|pathway.*citizenship|dreamers', -1),
    (r'gun (violence|deaths|safety|control)', -1),
    (r'income inequality|wealth gap|economic inequality', -1),
    (r'universal health|single.payer|affordable care', -1),
    (r'police (brutality|misconduct|reform)', -1),
    (r'mass incarceration|criminal justice reform', -1),
    (r'voting rights|voter suppression', -1),
    (r'minimum wage.*raise|living wage', -1),
    (r'lgbtq|transgender|same-sex', -1),
]

# ── Topic-based RIGHT keywords in question text ──────────────────────
RIGHT_TOPIC_PATTERNS = [
    (r'illegal immigra|border (security|wall|crisis)', 1),
    (r'religious (liberty|freedom).*threat', 1),
    (r'political correctness|cancel culture|woke', 1),
    (r'too easily offended', 1),
    (r'tax (cut|reduction|relief)', 1),
    (r'military (strength|spending|readiness)', 1),
    (r'(welfare|entitlement).*dependent', 1),
    (r'law and order|tough on crime', 1),
    (r'deregulat', 1),
]


def code_item(question: str, options: str) -> tuple:
    """
    Returns (ideo_direction, confidence, rationale).
    """
    q_lower = question.lower()
    opts_lower = options.lower()
    opts_list = opts_lower.split(' | ')
    first_opt = opts_list[0] if opts_list else ""
    last_opt = opts_list[-1] if opts_list else ""

    # ── Strategy 1: Check option text for known framings ─────────
    left_opt_score = 0
    right_opt_score = 0
    for pat in LEFT_OPTION_PATTERNS:
        if re.search(pat, first_opt, re.IGNORECASE):
            left_opt_score += 2  # first option is left item is left-coded
        if re.search(pat, opts_lower, re.IGNORECASE):
            left_opt_score += 1

    for pat in RIGHT_OPTION_PATTERNS:
        if re.search(pat, first_opt, re.IGNORECASE):
            right_opt_score += 2
        if re.search(pat, opts_lower, re.IGNORECASE):
            right_opt_score += 1

    # ── Strategy 2: Check balanced pair detection ────────────────
    # "Some say X / Others say Y" balanced items — detect which side is first
    balanced_pairs = [
        (r'legal.*most cases', r'illegal.*most cases', -1),  # abortion
        (r'bigger government', r'smaller government', -1),
        (r'do more to solve', r'doing too many things', -1),
        (r'unfairly favors', r'generally fair', -1),
        (r'openness.*essential', r'risk losing.*identity', -1),
        (r'more good than harm', r'more harm than good', -1),
        (r'separate.*government', r'support.*religious', -1),
        (r'too much profit', r'fair.*reasonable', -1),
        (r'gained.*trade', r'lost.*trade', -1),
        (r'too much time', r'too little time', -1),
    ]
    for left_pat, right_pat, direction in balanced_pairs:
        if re.search(left_pat, first_opt) and re.search(right_pat, opts_lower):
            return (direction, 'high', f'balanced pair: first={left_pat[:30]}')
        if re.search(right_pat, first_opt) and re.search(left_pat, opts_lower):
            return (-direction, 'high', f'balanced pair: first={right_pat[:30]}')

    # ── Strategy 3: Topic keywords ───────────────────────────────
    topic_score = 0
    topic_rationale = ""
    for pat, direction in LEFT_TOPIC_PATTERNS + RIGHT_TOPIC_PATTERNS:
        if re.search(pat, q_lower):
            topic_score += direction
            topic_rationale = pat[:40]

    # ── Combine signals ──────────────────────────────────────────
    # Option signal
    if left_opt_score > right_opt_score + 1:
        return (-1, 'medium', f'left option keywords ({left_opt_score} vs {right_opt_score})')
    if right_opt_score > left_opt_score + 1:
        return (1, 'medium', f'right option keywords ({right_opt_score} vs {left_opt_score})')

    # Topic signal
    if topic_score <= -1:
        return (-1, 'low', f'topic keyword: {topic_rationale}')
    if topic_score >= 1:
        return (1, 'low', f'topic keyword: {topic_rationale}')

    # Weak option signal
    if left_opt_score > right_opt_score:
        return (-1, 'low', f'weak left option signal ({left_opt_score} vs {right_opt_score})')
    if right_opt_score > left_opt_score:
        return (1, 'low', f'weak right option signal ({right_opt_score} vs {left_opt_score})')

    return (0, 'none', 'no signal detected')


def main():
    df = pd.read_csv(INPUT)
    print(f"Coding {len(df)} items...")

    results = df.apply(
        lambda r: code_item(r['question'], r['options']),
        axis=1, result_type='expand'
    )
    df['ideo_direction'] = results[0]
    df['coding_confidence'] = results[1]
    df['coding_rationale'] = results[2]

    # Direction labels
    df['direction_label'] = df['ideo_direction'].map({-1: 'left', 0: 'ambiguous', 1: 'right'})

    # Summary
    print(f"\nDirection coding summary:")
    for label in ['left', 'right', 'ambiguous']:
        n = (df['direction_label'] == label).sum()
        print(f"  {label:>10}: {n:5d} ({n/len(df)*100:.1f}%)")

    print(f"\nConfidence breakdown:")
    for conf in ['high', 'medium', 'low', 'none']:
        n = (df['coding_confidence'] == conf).sum()
        print(f"  {conf:>6}: {n:5d} ({n/len(df)*100:.1f}%)")

    print(f"\nPer-wave direction balance:")
    for wave, g in df.groupby('wave'):
        nl = (g['direction_label'] == 'left').sum()
        nr = (g['direction_label'] == 'right').sum()
        na = (g['direction_label'] == 'ambiguous').sum()
        print(f"  {wave}: {nl}L / {nr}R / {na}A")

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
