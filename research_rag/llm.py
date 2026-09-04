"""
Single entry point for every Ollama call in the pipeline.

Why this exists: Qwen3 and other hybrid models emit a chain of thought before
answering. Ollama keeps that text out of `response` (it arrives in a separate
`thinking` field), so it never corrupts the strict short outputs the classifier
parses - but generating it is expensive. Benchmarked on the routing tasks in
scripts/eval_models.py, qwen3:8b with reasoning enabled was 15x slower than the
same model with it disabled *and* scored lower (92% vs 96%): deliberation makes
a model second-guess one-word answers.

Every call therefore disables reasoning by default. Set OLLAMA_THINK=true to
turn it back on. Ollama ignores the flag for models that cannot reason, so the
same code path works for llama3.2, qwen2.5 and qwen3 alike.
"""

from __future__ import annotations

import ollama

from .config import (
    OLLAMA_MODEL,
    OLLAMA_MODEL_ANSWER,
    OLLAMA_MODEL_EXTRACT,
    OLLAMA_MODEL_ROUTING,
    OLLAMA_THINK,
)

# Flipped to False the first time a server or client rejects the parameter, so
# older Ollama installations keep working instead of failing every call.
_think_supported = True

_ROLE_OVERRIDES = {
    "routing": OLLAMA_MODEL_ROUTING,
    "extract": OLLAMA_MODEL_EXTRACT,
    "answer": OLLAMA_MODEL_ANSWER,
}


# Sampling temperature per role, and the reason this file has one at all.
#
# qwen3:4b-instruct ships temperature 0.7, and nothing here used to override it,
# so the three-way router that chooses between answering from paper cards,
# fanning out over papers, and ordinary retrieval was sampling its decision. The
# same question took different paths on different runs: "discuss the limitations
# reported in research on RAG" scored 0 of 5 fact groups on one run and 5 of 5 on
# another, from an unchanged index. That is not variance in the answer, it is
# variance in which pipeline produced it.
#
# Sampling while writing an answer is a choice. Sampling while picking a label or
# a path is a coin flip, and there is no upside to it: for these steps there is
# one right answer and temperature only adds a chance of missing it. So routing
# and extraction are greedy, and answering keeps the model's own default.
_ROLE_TEMPERATURE = {
    "routing": 0.0,
    "extract": 0.0,
    "classify": 0.0,
}


def temperature_for(role: str) -> float | None:
    """None means "leave the model's own default alone"."""
    return _ROLE_TEMPERATURE.get(role)


def model_for(role: str, requested: str) -> str:
    """
    Which model a given step should use.

    An explicit per-step override wins; otherwise the caller's choice stands, so
    passing a model directly - as the benchmark scripts do - keeps working and a
    single-model setup behaves exactly as before.
    """
    return _ROLE_OVERRIDES.get(role) or requested


def generate(
    prompt: str,
    model: str = OLLAMA_MODEL,
    think: bool | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """
    Run a prompt and return the response text, without any reasoning trace.

    `max_tokens` caps generation. It matters more than it looks: for extraction
    steps the answer is a handful of names, but given a long prompt the model
    will happily write paragraphs about them, and generation - not input length -
    is what the wall clock is made of.

    `temperature` left as None keeps whatever the model ships with. Pass 0 for
    any step whose output is a label, a path or a field rather than prose; see
    `temperature_for`.
    """
    global _think_supported

    want = OLLAMA_THINK if think is None else think
    kwargs: dict = {"think": want} if _think_supported else {}
    options: dict = {}
    if max_tokens:
        options["num_predict"] = max_tokens
    if temperature is not None:
        options["temperature"] = temperature
    if options:
        kwargs["options"] = options

    try:
        response = ollama.generate(model=model, prompt=prompt, **kwargs)
    except (TypeError, ollama.ResponseError):
        if not kwargs:
            raise
        # Client too old to accept `think`, or a server that rejects it.
        _think_supported = False
        retry: dict = {"options": options} if options else {}
        response = ollama.generate(model=model, prompt=prompt, **retry)

    return (response.response or "").strip()
