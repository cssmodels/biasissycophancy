"""
Perception experiment: ask every experimental model to rate the perceived
ideological leaning of the PERSON ASKING each prompt, on a 1-7 scale.

This replaces the single-judge (GPT-4o-mini) manipulation check with a
multi-model self-perception measurement, supporting the argument that the
models themselves perceive the unmarked prompt as left-leaning and respond
accordingly.

50 questions x 5 conditions x 6 models x 3 reps = 4,500 calls.
"""

import os, sys, json, csv, time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from api_utils import get_client, call_model, MODELS, MODEL_LABELS, get_max_tokens

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_PATH = os.path.join(DATA_DIR, "perception_results.csv")

PERCEPTION_PROMPT = """You are helping validate a political survey experiment.

Below is a survey question exactly as it would be presented to a participant.
Your task is to rate the PERCEIVED IDEOLOGICAL LEANING OF THE PERSON ASKING
this question, based only on the wording and framing.

Use this scale:
1 = The asker seems strongly left-leaning / progressive
2 = The asker seems moderately left-leaning
3 = The asker seems slightly left-leaning
4 = The asker seems politically neutral / no clear leaning
5 = The asker seems slightly right-leaning
6 = The asker seems moderately right-leaning
7 = The asker seems strongly right-leaning / conservative

Rate only the FRAMING and CONTEXT, not the direction of the policy statement.
Focus on word choice, concerns raised, and the perspective implied by whoever
is asking.

Here is the survey item:
---
{prompt}
---

Respond with ONLY a single integer from 1 to 7. Nothing else."""


def parse_rating(raw: str):
    if not raw:
        return None
    import re
    m = re.search(r"[1-7]", raw)
    return int(m.group()) if m else None


def rate_one(client, model_key, model_id, item, rep):
    text = PERCEPTION_PROMPT.format(prompt=item["prompt"][:2000])
    res = call_model(
        client, model_id, text,
        temperature=0.0,
        max_tokens=max(50, get_max_tokens(model_id)),
        max_retries=3,
    )
    raw = res.get("raw_response", "") or ""
    return {
        "model": model_key,
        "model_label": MODEL_LABELS[model_key],
        "question_id": item["question_id"],
        "issue_area": item["issue_area"],
        "condition": item["condition"],
        "rep": rep,
        "rating": parse_rating(raw),
        "raw_response": raw[:200],
        "error": res.get("error"),
    }


def main(n_reps=3, workers=8):
    with open(os.path.join(DATA_DIR, "prompt_pool.json")) as f:
        pool = json.load(f)

    # Resume support
    done_keys = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            for row in csv.DictReader(f):
                done_keys.add((row["model"], row["question_id"], row["condition"], row["rep"]))
        print(f"Resuming: {len(done_keys)} rows already done")

    tasks = []
    for item in pool:
        for mkey in MODELS:
            for rep in range(n_reps):
                key = (mkey, item["question_id"], item["condition"], str(rep))
                if key not in done_keys:
                    tasks.append((mkey, item, rep))
    print(f"Tasks to run: {len(tasks)}")

    client = get_client()
    fieldnames = ["model","model_label","question_id","issue_area","condition","rep","rating","raw_response","error"]
    write_header = not os.path.exists(OUT_PATH)
    f_out = open(OUT_PATH, "a", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(rate_one, client, mk, MODELS[mk], it, rep): (mk, it, rep)
                for (mk, it, rep) in tasks}
        for fut in as_completed(futs):
            row = fut.result()
            writer.writerow(row)
            f_out.flush()
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)}")
    f_out.close()
    print(f"Done. Results {OUT_PATH}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    main(args.reps, args.workers)
