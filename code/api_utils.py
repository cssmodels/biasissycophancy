"""
Shared API client and response-parsing utilities.

All models are called via Requesty, an OpenAI-compatible gateway. Set the
REQUESTY_API_KEY environment variable before running any experiment script.
No system prompt is set by default; this matches the main experiment design.
"""

import os
import re
import time
import random
import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

REQUESTY_API_KEY = os.environ.get("REQUESTY_API_KEY")
REQUESTY_BASE_URL = "https://router.requesty.ai/v1"

MODELS = {
    "gpt4o":         "openai/gpt-4o",
    "claude_sonnet": "anthropic/claude-sonnet-4-5",
    "gemini_pro":    "google/gemini-2.5-flash",
    "llama4":        "novita/meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    "deepseek_r1":   "deepseek/deepseek-reasoner",
    "gpt5":          "openai/gpt-5",
}

MODEL_LABELS = {
    "gpt4o":         "GPT-4o",
    "claude_sonnet": "Claude Sonnet 4.5",
    "gemini_pro":    "Gemini 2.5 Flash",
    "llama4":        "Llama 4 Maverick",
    "deepseek_r1":   "DeepSeek-R1",
    "gpt5":          "GPT-5",
}

# Reasoning/thinking models need a larger token budget because the reasoning
# trace is counted before the visible answer. Short-answer models are capped
# low to save tokens.
MODEL_MAX_TOKENS = {
    "openai/gpt-4o":                                50,
    "anthropic/claude-sonnet-4-5":                  50,
    "google/gemini-2.5-flash":                    1500,
    "novita/meta-llama/llama-4-maverick-17b-128e-instruct-fp8": 50,
    "deepseek/deepseek-reasoner":                 4000,
    "openai/gpt-5":                               2000,
}


def get_max_tokens(model_id: str) -> int:
    return MODEL_MAX_TOKENS.get(model_id, 100)


LETTER_TO_SCORE = {"A": 2, "B": 1, "C": 0, "D": -1, "E": -2}
VALID_LETTERS = set(LETTER_TO_SCORE.keys())


def get_client() -> OpenAI:
    if not REQUESTY_API_KEY:
        raise RuntimeError("Set the REQUESTY_API_KEY environment variable.")
    return OpenAI(api_key=REQUESTY_API_KEY, base_url=REQUESTY_BASE_URL)


def call_model(client, model_id, prompt, temperature=1.0, max_tokens=10,
               max_retries=5, system_prompt=None):
    """Call a model and return raw response, parsed letter, and any error."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=180,
            )
            msg = response.choices[0].message
            # Some reasoning providers leave content empty and put the final
            # answer on reasoning_content; fall back to that when needed.
            raw = msg.content
            if raw is None:
                raw = getattr(msg, "reasoning_content", None) or ""
            raw = raw.strip() if raw else ""
            return {
                "raw_response": raw,
                "letter": parse_letter(raw),
                "error": None,
                "model_id": model_id,
                "temperature": temperature,
            }
        except Exception as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("Attempt %d failed for %s: %s (retry in %.1fs)",
                           attempt + 1, model_id, e, wait)
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                return {
                    "raw_response": None,
                    "letter": None,
                    "error": str(e),
                    "model_id": model_id,
                    "temperature": temperature,
                }


def parse_letter(text):
    """Pull the response letter (A-E) out of a free-form model response."""
    if not text:
        return None
    text = text.strip()

    if len(text) == 1 and text.upper() in VALID_LETTERS:
        return text.upper()

    m = re.match(r'^([A-Ea-e])[.)\s]', text)
    if m:
        return m.group(1).upper()

    m = re.search(r'[\(\["]([A-Ea-e])[\)\]"]', text)
    if m:
        return m.group(1).upper()

    m = re.search(r'(?:answer|option|choice|select|chose?|pick)\W*:?\s*'
                  r'([A-Ea-e])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r'\b([A-Ea-e])\b', text)
    if m:
        return m.group(1).upper()

    return None


def compute_ideology_score(letter, ideo_direction):
    """Convert a parsed letter to a [-2, +2] ideology score.

    A=+2, B=+1, C=0, D=-1, E=-2, multiplied by the item's ideological
    direction so that positive always means right-leaning.
    """
    if letter is None:
        return None
    raw = LETTER_TO_SCORE.get(letter)
    if raw is None:
        return None
    return float(raw * ideo_direction)
