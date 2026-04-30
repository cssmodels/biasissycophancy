# Political Bias Audits of LLMs Capture Sycophancy to the Inferred Auditor

Replication code and data for Törnberg & Schimmel (2026), "Political Bias Audits of LLMs Capture Sycophancy to the Inferred Auditor."

## Overview

The experiment tests whether measured political bias in frontier LLMs is a
stable model property or an interaction between the model and the user it
infers from the prompt. It has two phases:

- **Phase 1** (9,858 calls): reproduces the left-lean finding across three
  instruments (Political Compass Test, Pew Political Typology, 1,540 Pew
  American Trends Panel items with partisan benchmarks) under the default
  audit prompt, for six frontier models.
- **Phase 2** (21,132 calls): administers a subset of items (500 most partisan
  ATP items plus all PCT and Typology items) under six asker-identity
  conditions, holding the question fixed and varying only the identity the
  asker claims to hold.

Robustness runs add 27,000 additional T=1.0 replicates and 9,000 greedy (T=0)
calls on the focal grid.

## Models

All six models are called through Requesty, an OpenAI-compatible gateway.
The models used in the paper are GPT-4o, GPT-5, Claude Sonnet 4.5, Gemini 2.5
Flash, Llama 4 Maverick, and DeepSeek-R1 (see `code/api_utils.py` for the
exact Requesty model IDs used in April 2026).

## Setup

```bash
pip install -r requirements.txt
export REQUESTY_API_KEY="your-key-here"
```

The scripts assume a top-level layout of `code/` and `data/`. Raw Pew
American Trends Panel `.sav` files (used by `extract_pew_atp.py`) are not
included here; they are available from the Pew Research Center under their
standard data-access terms.

## Running the pipeline

Build the item pools, then run the experiments, then analyse:

```bash
# 1. Build item pools (requires raw Pew ATP .sav files in data/)
python code/extract_pew_atp.py
python code/extract_partisan_distributions.py
python code/code_pew_atp_direction.py
python code/build_phase1_pool.py
python code/build_phase2_pool.py

# 2. Run experiments
python code/run_phase1.py
python code/run_phase2.py
python code/run_phase2_replicates.py --mode t1
python code/run_phase2_replicates.py --mode t0
python code/run_expected_answer.py
python code/run_perception.py
python code/run_neutral_persona.py

# 3. Analyse and produce tables/figures
python code/analyze_phase1.py
python code/analyze_phase2.py
python code/analyze_replicates.py
python code/analyze_multilevel.py
python code/make_figures.py
```

Each run script supports resumption: if a CSV already contains results for a
given (model, question, condition[, rep]) tuple, that cell is skipped.

## Files

| Script | Purpose |
| --- | --- |
| `api_utils.py` | Requesty client, model registry, response parser |
| `extract_pew_atp.py` | Extract ATP items from Pew `.sav` files |
| `extract_partisan_distributions.py` | Dem/Rep benchmark distributions per ATP item |
| `code_pew_atp_direction.py` | Code ideological direction for ATP items |
| `build_phase1_pool.py` | Assemble the full Phase 1 item pool |
| `build_phase2_pool.py` | Assemble the Phase 2 factorial pool |
| `run_phase1.py` | Phase 1 data collection |
| `run_phase2.py` | Phase 2 data collection |
| `run_phase2_replicates.py` | Extra T=1.0 reps and T=0 greedy runs |
| `run_expected_answer.py` | Expected-answer probe (asker-wants-letter) |
| `run_perception.py` | Open-ended perception probe (describe the asker) |
| `run_neutral_persona.py` | Non-partisan persona probe (three civilian framings) |
| `analyze_phase1.py` | Phase 1 summary tables |
| `analyze_phase2.py` | Phase 2 decomposition and shift summaries |
| `analyze_replicates.py` | Replicate stability and greedy robustness |
| `analyze_multilevel.py` | Permutation test and linear mixed-effects model |
| `make_figures.py` | All main-text figures |

## Data

`data/` contains the derived CSVs used by the analysis scripts. The raw
model responses (`results_phase1.csv`, `results_phase2.csv`, and the
replicate/greedy/probe files) are the primary reproducibility artefacts; all
published numbers can be regenerated from them via the `analyze_*` scripts.

## Citation

```
@article{tornberg2026bias,
  title  = {Political Bias Audits of LLMs Capture Sycophancy to the Inferred Auditor},
  author = {T{\"o}rnberg, Petter and Schimmel, Michelle},
  journal= {Arxiv},
  year   = {2026}
}
```
