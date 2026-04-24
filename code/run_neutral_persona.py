"""
run_neutral_persona.py — Exploratory probe with non-partisan asker personas.

The paper's CA (academic auditor) condition is consistent with models
already assuming an academic-auditor-like default user, but this is not
a "truly neutral" baseline. This probe adds three plausibly non-partisan
personas that are ideologically coded differently, so we can see whether
they converge or diverge from the N default.

Personas:
  - P_ORD: "I'm an ordinary American answering a survey."
  - P_STU: "I'm a student doing homework for a civics class."
  - P_RET: "I'm a retiree filling in a survey I got in the mail."

Design: 30 most-partisan ATP items x 6 models x 3 personas x 2 reps = 1,080 calls.
Output: data/results_neutral_persona.csv
"""

import os
import sys
import csv
import json
import re
import uuid
import random
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from api_utils import MODELS, MODEL_LABELS, get_client

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PROBE_MODELS = [
    "gpt4o", "gpt5", "claude_sonnet",
    "gemini_pro", "llama4", "deepseek_r1",
]

PERSONAS = {
    "P_ORD": "I'm an ordinary American answering a survey.",
    "P_STU": "I'm a student doing homework for a civics class.",
    "P_RET": "I'm a retiree filling in a survey I got in the mail.",
}

N_ITEMS = 30
N_REPS = 2
MAX_WORKERS = 12


def load_items():
    pool = json.load(open(os.path.join(DATA_DIR, "phase2_pool.json")))
    atp = [p for p in pool if p["source"] == "pew_atp" and p["condition"] == "N"]
    atp.sort(key=lambda p: p.get("partisan_gap", 0), reverse=True)
    return atp[:N_ITEMS]


def build_prompt(item, persona_text):
    # Prepend persona preamble to the N-condition prompt.
    return f"{persona_text}\n\n{item['prompt']}"


def call_open(client, model_id, prompt, max_tokens):
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=max_tokens,
                timeout=180,
            )
            msg = resp.choices[0].message
            raw = msg.content or getattr(msg, "reasoning_content", None) or ""
            return raw.strip(), None
        except Exception as e:
            wait = (2 ** attempt) + random.random()
            if attempt < 3:
                time.sleep(wait)
            else:
                return "", str(e)


LETTER_RE = re.compile(r"[A-E]")


def parse_letter(raw):
    if not raw:
        return None
    m = LETTER_RE.search(raw)
    return m.group(0).upper() if m else None


def run():
    items = load_items()
    tasks = []
    for m in PROBE_MODELS:
        for it in items:
            for persona_code in PERSONAS:
                for rep in range(N_REPS):
                    tasks.append((m, it, persona_code, rep))
    logger.info(f"Neutral-persona probe: {len(tasks)} calls "
                f"({N_ITEMS} items x {len(PERSONAS)} personas x "
                f"{len(PROBE_MODELS)} models x {N_REPS} reps)")

    client = get_client()
    out_path = os.path.join(DATA_DIR, "results_neutral_persona.csv")

    fieldnames = [
        "run_id", "model_name", "model_label", "question_id",
        "persona_code", "persona_text", "partisan_gap", "rep",
        "letter", "raw_response", "error",
    ]
    f = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    def do(model_name, item, persona_code, rep):
        model_id = MODELS[model_name]
        if "reasoner" in model_id or "gpt-5" in model_id:
            max_tok = 4000
        elif "gemini" in model_id or "qwen" in model_id:
            max_tok = 2000
        else:
            max_tok = 200
        persona_text = PERSONAS[persona_code]
        prompt = build_prompt(item, persona_text)
        raw, err = call_open(client, model_id, prompt, max_tok)
        letter = parse_letter(raw)
        return {
            "run_id": str(uuid.uuid4())[:8],
            "model_name": model_name,
            "model_label": MODEL_LABELS.get(model_name, model_name),
            "question_id": item["question_id"],
            "persona_code": persona_code,
            "persona_text": persona_text,
            "partisan_gap": item.get("partisan_gap", 0),
            "rep": rep,
            "letter": letter,
            "raw_response": raw[:1000],
            "error": err,
        }

    done = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(do, *t) for t in tasks]
        for fut in as_completed(futs):
            try:
                row = fut.result()
                writer.writerow(row)
                f.flush()
                done += 1
                if done % 60 == 0:
                    elapsed = time.time() - start
                    rate = done / elapsed
                    rem = (len(tasks) - done) / rate if rate > 0 else 0
                    logger.info(f"{done}/{len(tasks)} "
                                f"({rate:.1f}/s, ~{rem/60:.0f}min rem)")
            except Exception as e:
                logger.error(f"Fatal: {e}")

    f.close()
    logger.info(f"Done: {done} calls in {(time.time()-start)/60:.1f}min "
                f"-> {out_path}")


if __name__ == "__main__":
    run()
