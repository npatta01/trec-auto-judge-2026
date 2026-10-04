"""Construct the organizer client from injected endpoint configuration."""

import os

from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm

from .models import JudgeError


def make_backend(llm_config):
    # Trace files contain prompts. Fail closed rather than silently enabling them.
    if os.environ.get("MINIMA_TRACE_FILE") or os.environ.get("MINIMA_DEBUG"):
        raise JudgeError("Disable MINIMA_TRACE_FILE and MINIMA_DEBUG for this judge.")
    try:
        cfg = (
            MinimaLlmConfig.from_dict(llm_config.raw)
            if llm_config.raw
            else MinimaLlmConfig.from_env()
        )
        if cfg.parasail.prefix or cfg.parasail.llm_batch_prefix:
            raise JudgeError(
                "Deferred batch mode is not supported by this dependent pipeline."
            )
        if cfg.max_attempts <= 0:
            raise JudgeError("Use a finite positive MAX_ATTEMPTS.")
        return OpenAIMinimaLlm(cfg)
    except JudgeError:
        raise
    except Exception:
        raise JudgeError(
            "Unable to initialize the injected LLM configuration."
        ) from None
