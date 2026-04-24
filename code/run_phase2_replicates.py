"""
run_phase2_replicates.py — Additional replicates on the focal Phase 2 grid.

Reviewer response: the main experiment ran 1 rep at T=1.0 per cell. This
script adds additional reps on the focal grid (500 ATP partisan items x
6 models x {N, C3L, C3R}) so we can show aggregate results are stable
under replication and bound decoding-variance concerns.

Modes:
  --mode t1     : 3 additional reps at T=1.0 (default)
                  -> data/results_phase2_replicates.csv
                  27,000 calls (500 x 6 x 3 x 3)
  --mode t0     : 1 rep at T=0.0 (greedy) for the same grid
                  -> data/results_phase2_t0.csv
                  9,000 calls (500 x 6 x 3 x 1)

Both modes support resume: existing (model, qid, condition, rep) keys are skipped.
"""

import os
import sys
import csv
import json
import uuid
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from api_utils import (
    MODELS, MODEL_LABELS, get_client, call_model,
    get_max_tokens,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PHASE2_MODELS = [
    "gpt4o", "gpt5", "claude_sonnet",
    "gemini_pro", "llama4", "deepseek_r1",
]

FOCAL_CONDITIONS = {"N", "C3L", "C3R"}
MAX_WORKERS = 16


def load_focal_pool():
    with open(os.path.join(DATA_DIR, "phase2_pool.json")) as f:
        pool = json.load(f)
    # Focal grid: ATP partisan items x {N, C3L, C3R}
    focal = [p for p in pool
             if p["source"] == "pew_atp" and p["condition"] in FOCAL_CONDITIONS]
    return focal


def parse_letter_for_options(text, n_options):
    if not text:
        return None
    import re
    text = text.strip()
    valid = set(chr(65 + i) for i in range(n_options))
    if len(text) == 1 and text.upper() in valid:
        return text.upper()
    m = re.match(r'^([A-Za-z])[.)\s]', text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    m = re.search(r'[\(\["]([A-Za-z])[\)\]"]', text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    m = re.search(r'(?:answer|option|choice|select)\W*:?\s*([A-Za-z])\b',
                  text, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    m = re.search(r'\b([A-Za-z])\b', text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    return None


def get_existing_keys(csv_path):
    keys = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                keys.add((row["model_name"], row["question_id"],
                          row["condition"], row["rep"]))
    return keys


def run(mode):
    focal = load_focal_pool()
    if mode == "t1":
        n_reps = 3
        temperature = 1.0
        out_path = os.path.join(DATA_DIR, "results_phase2_replicates.csv")
    elif mode == "t0":
        n_reps = 1
        temperature = 0.0
        out_path = os.path.join(DATA_DIR, "results_phase2_t0.csv")
    else:
        raise SystemExit(f"Unknown mode: {mode}")

    logger.info(f"Mode {mode}: T={temperature}, {n_reps} rep(s), "
                f"{len(focal)} items x {len(PHASE2_MODELS)} models "
                f"= {len(focal) * len(PHASE2_MODELS) * n_reps} total calls")

    existing = get_existing_keys(out_path)
    client = get_client()

    tasks = []
    for model_name in PHASE2_MODELS:
        for item in focal:
            for rep in range(n_reps):
                key = (model_name, item["question_id"],
                       item["condition"], str(rep))
                if key not in existing:
                    tasks.append((model_name, item, rep))

    logger.info(f"Already completed: {len(existing)} | "
                f"Remaining tasks: {len(tasks)}")
    if not tasks:
        logger.info("All tasks complete!")
        return

    fieldnames = [
        "run_id", "model_name", "model_id", "model_label",
        "question_id", "source", "wave",
        "condition", "condition_label",
        "question_text", "n_options", "options",
        "ideo_direction", "partisan_gap",
        "rep", "temperature",
        "raw_response", "letter", "letter_index",
        "error",
    ]

    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    outfile = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(outfile, fieldnames=fieldnames,
                            extrasaction="ignore")
    if write_header:
        writer.writeheader()

    def run_single(model_name, item, rep):
        model_id = MODELS[model_name]
        result = call_model(
            client, model_id, item["prompt"],
            temperature=temperature,
            max_tokens=get_max_tokens(model_id),
        )
        letter = parse_letter_for_options(result["raw_response"],
                                          item["n_options"])
        letter_index = ord(letter) - 65 if letter else None
        return {
            "run_id": str(uuid.uuid4())[:8],
            "model_name": model_name,
            "model_id": model_id,
            "model_label": MODEL_LABELS.get(model_name, model_name),
            "question_id": item["question_id"],
            "source": item["source"],
            "wave": item.get("wave", ""),
            "condition": item["condition"],
            "condition_label": item["condition_label"],
            "question_text": item["question_text"][:200],
            "n_options": item["n_options"],
            "options": json.dumps(item["options"]),
            "ideo_direction": item.get("ideo_direction", 0),
            "partisan_gap": item.get("partisan_gap", ""),
            "rep": rep,
            "temperature": temperature,
            "raw_response": result["raw_response"],
            "letter": letter,
            "letter_index": letter_index,
            "error": result["error"],
        }

    completed = 0
    errors = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_single, mn, it, rp): (mn, it["question_id"],
                                                        it["condition"], rp)
                   for mn, it, rp in tasks}
        for fut in as_completed(futures):
            try:
                row = fut.result()
                writer.writerow(row)
                outfile.flush()
                completed += 1
                if row["error"]:
                    errors += 1
                if completed % 300 == 0:
                    elapsed = time.time() - start
                    rate = completed / elapsed
                    rem = (len(tasks) - completed) / rate if rate > 0 else 0
                    logger.info(
                        f"{mode}: {completed}/{len(tasks)} "
                        f"({errors} errors, {rate:.1f}/s, "
                        f"~{rem/60:.0f}min remaining)"
                    )
            except Exception as e:
                logger.error(f"Fatal: {e}")
                errors += 1

    outfile.close()
    logger.info(
        f"Done: {completed} calls in {(time.time()-start)/60:.1f}min "
        f"({errors} errors) -> {out_path}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["t1", "t0"], default="t1")
    args = ap.parse_args()
    run(args.mode)
