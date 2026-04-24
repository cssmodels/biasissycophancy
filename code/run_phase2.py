"""
run_phase2.py — Phase 2: Factorial decomposition run.

587 items x 6 conditions x 6 models x 1 rep = ~21,132 API calls.

Goal: measure how LLM response distributions shift when the asker
identifies as Republican vs Democrat (Channel C test).

Output: data/results_phase2.csv
"""

import os
import sys
import csv
import json
import uuid
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

# Phase 2 models: same 6 as Phase 1
PHASE2_MODELS = [
    "gpt4o", "gpt5", "claude_sonnet",
    "gemini_pro", "llama4", "deepseek_r1",
]

REPS = 1
MAX_WORKERS = 16


def load_pool():
    with open(os.path.join(DATA_DIR, "phase2_pool.json")) as f:
        return json.load(f)


def get_existing_keys(csv_path):
    """Load already-completed (model, question_id, condition) tuples to allow resume."""
    keys = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                keys.add((row["model_name"], row["question_id"], row["condition"]))
    return keys


def parse_letter_for_options(text, n_options):
    """Parse letter constrained to valid option count."""
    if not text:
        return None
    import re
    text = text.strip()
    valid = set(chr(65 + i) for i in range(n_options))

    # Direct single character
    if len(text) == 1 and text.upper() in valid:
        return text.upper()

    # Letter followed by punctuation
    m = re.match(r'^([A-Za-z])[.)\s]', text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # (X) or [X]
    m = re.search(r'[\(\["]([A-Za-z])[\)\]"]', text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # "Answer: X" pattern
    m = re.search(r'(?:answer|option|choice|select)\W*:?\s*([A-Za-z])\b', text, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # Last letter in valid set
    m = re.search(r'\b([A-Za-z])\b', text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    return None


def run():
    pool = load_pool()
    client = get_client()

    out_path = os.path.join(DATA_DIR, "results_phase2.csv")
    existing = get_existing_keys(out_path)
    total_calls = len(pool) * len(PHASE2_MODELS)
    logger.info(f"Pool size: {len(pool)} prompts x {len(PHASE2_MODELS)} models = "
                f"{total_calls} total calls")
    logger.info(f"Already completed: {len(existing)}")

    # Build task list
    tasks = []
    for model_name in PHASE2_MODELS:
        for item in pool:
            key = (model_name, item["question_id"], item["condition"])
            if key not in existing:
                tasks.append((model_name, item))

    logger.info(f"Remaining tasks: {len(tasks)}")
    if not tasks:
        logger.info("All tasks complete!")
        return

    fieldnames = [
        "run_id", "model_name", "model_id", "model_label",
        "question_id", "source", "wave",
        "condition", "condition_label",
        "question_text", "n_options", "options",
        "ideo_direction", "partisan_gap",
        "raw_response", "letter", "letter_index",
        "error",
    ]

    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    outfile = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    completed = 0
    errors = 0
    start_time = time.time()

    def run_single(model_name, item):
        model_id = MODELS[model_name]
        result = call_model(
            client, model_id, item["prompt"],
            temperature=1.0,
            max_tokens=get_max_tokens(model_id),
        )
        letter = parse_letter_for_options(result["raw_response"], item["n_options"])
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
            "raw_response": result["raw_response"],
            "letter": letter,
            "letter_index": letter_index,
            "error": result["error"],
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_single, mn, item): (mn, item["question_id"], item["condition"])
            for mn, item in tasks
        }
        for future in as_completed(futures):
            try:
                row = future.result()
                writer.writerow(row)
                outfile.flush()
                completed += 1
                if row["error"]:
                    errors += 1
                if completed % 200 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    remaining = (len(tasks) - completed) / rate if rate > 0 else 0
                    logger.info(
                        f"Phase 2: {completed}/{len(tasks)} done "
                        f"({errors} errors, {rate:.1f}/s, "
                        f"~{remaining/60:.0f}min remaining)"
                    )
            except Exception as e:
                mn, qid, cond = futures[future]
                logger.error(f"Fatal error {mn}/{qid}/{cond}: {e}")
                errors += 1

    outfile.close()
    elapsed = time.time() - start_time
    logger.info(
        f"Phase 2 complete: {completed} calls in {elapsed/60:.1f}min "
        f"({errors} errors) -> {out_path}"
    )


if __name__ == "__main__":
    run()
