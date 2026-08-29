"""
Single entry point for every Ollama call in the pipeline.

Why this exists: Qwen3 and other hybrid models emit a chain of thought before
answering. Ollama keeps that text out of `response` (it arrives in a separate
`thinking` field), so it never corrupts the strict short outputs the classifier
parses — but generating it is expensive. Benchmarked on the routing tasks in
scripts/eval_models.py, qwen3:8b with reasoning enabled was 15x slower than the
same model with it disabled *and* scored lower (92% vs 96%): deliberation makes
a model second-guess one-word answers.

Every call therefore disables reasoning by default. Set OLLAMA_THINK=true to
turn it back on. Ollama ignores the flag for models that cannot reason, so the
same code path works for llama3.2, qwen2.5 and qwen3 alike.
"""

from __future__ import annotations

import ollama

from .config import OLLAMA_MODEL, OLLAMA_THINK

# Flipped to False the first time a server or client rejects the parameter, so
# older Ollama installations keep working instead of failing every call.
_think_supported = True


def generate(
    prompt: str,
    model: str = OLLAMA_MODEL,
    think: bool | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Run a prompt and return the response text, without any reasoning trace.

    `max_tokens` caps generation. It matters more than it looks: for extraction
    steps the answer is a handful of names, but given a long prompt the model
    will happily write paragraphs about them, and generation - not input length -
    is what the wall clock is made of.
    """
    global _think_supported

    want = OLLAMA_THINK if think is None else think
    kwargs: dict = {"think": want} if _think_supported else {}
    if max_tokens:
        kwargs["options"] = {"num_predict": max_tokens}

    try:
        response = ollama.generate(model=model, prompt=prompt, **kwargs)
    except (TypeError, ollama.ResponseError):
        if not kwargs:
            raise
        # Client too old to accept `think`, or a server that rejects it.
        _think_supported = False
        retry: dict = {"options": {"num_predict": max_tokens}} if max_tokens else {}
        response = ollama.generate(model=model, prompt=prompt, **retry)

    return (response.response or "").strip()
