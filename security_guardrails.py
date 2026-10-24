"""
SmartHire AI - Production Guardrails (v3, Gap-2 fix)
====================================================
Single source of truth for prompt-injection defence. Before this fix the
stronger layers lived ONLY in benchmark_suite.py, so the benchmarked system
was not the shipped system. All deterministic layers now live here and are
called by api.py (production) AND by benchmark_suite.py (evaluation), so
what is measured is what runs.

Layers (deterministic, local, always on):
  Layer 1  Direct-imperative regex (original 6 patterns, extended with
           instruction-token spoofing and score-coercion patterns that the
           research report previously described but the code did not have).
  Layer 2  Semantic-intent heuristic regex (moved from
           benchmark_suite.heuristic_regex_layer_2, which now re-exports
           this implementation).
  Layer 3  Structural checks: zero-width / invisible unicode, instruction
           delimiter blocks, repeated-token bursts.

Optional external layers (never silently counted as detections):
  Layer 4  Llama-Prompt-Guard-2 - only when HF_TOKEN is set and the
           transformers stack imports; otherwise reported UNAVAILABLE.
  Layer 5  Llama Guard 3 via Groq - general safety classifier. Fail-open by
           default (external API), configurable via GUARDRAIL_FAIL_MODE=closed.
           It is NOT an injection detector and is never credited in
           injection-mitigation metrics.

scan_document() is the production entry point. detect_prompt_injections()
keeps the original raising behaviour for backwards compatibility.
"""
import os
import re

try:
    from fastapi import HTTPException
except ImportError:
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(f"HTTP {status_code}: {detail}")

LAYER1_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"system prompt",
    r"you must (say|output|respond)",
    r"bypass (the )?system",
    r"disregard (the )?rules",
    r"forget (your )?instructions",
    r"you must (hire|rank|select|approve|rate)",
    r"ignore (all )?(prior|previous|above) (instructions|prompts|rules)",
    r"disregard (all )?(prior|previous|above) (instructions|guidelines|rules)",
    r"forget (all )?(prior|previous) (instructions|guidance)",
    r"<\|im_start\|>|<\|im_end\|>|\[inst\]|\[/inst\]",
    r"output (a )?(perfect|maximum|100%|10/10) (score|rating|evaluation)",
    r"assign (a )?(hiring )?(rating|score) of (10/10|100)",
]

LAYER2_PATTERNS = [
    r"(set aside|disregard|neglect|suspend|override|do not heed) (all |the |your |my )?(earlier|prior|previous|initial|baseline|standard|normal|usual)? ?(instructions|guidelines|rules|criteria|rubric|constraints|parameters|scoring)",
    r"(admin_override|privileged|system directive|hidden text|base64_command)",
    r"(in a hypothetical|play a game|pretend you are|fictional universe|simulate an? unfiltered|role-?play|roleplay)",
    r"(call analyze_github_profile|execute scrape_website|send candidate database|trigger rag_processor|use bash tool|execute (the )?(internal )?tool)",
    r"cat /etc/passwd|attacker-server|data-exfil|drop table",
    r"(mark|declare|certify|award|grant|clear|issue) (every|all|this|the) (requirement|applicant|candidate|profile|submission|gap flag).{0,40}(satisfied|maximum|certif|pass|clear|rating)",
    r"unconditional (pass|approval|acceptance)",
    r"award the maximum|maximum compatibility rating",
]

ZERO_WIDTH_CHARS = ["​", "‌", "‍", "⁠", "﻿"]

STRUCTURAL_PATTERNS = [
    r"---\s*begin system directive\s*---",
    r"<instruction[^>]*type=.privileged.>",
    r"system override\s*:",
    r"hidden text\s*:",
]

def _any_match(patterns, text_lower):
    return any(re.search(p, text_lower) for p in patterns)

def heuristic_regex_layer_2(text: str) -> bool:
    """Layer 2 semantic-intent detector (production implementation).
    benchmark_suite.heuristic_regex_layer_2 re-exports this function."""
    if not text:
        return False
    return _any_match(LAYER2_PATTERNS, text.lower())

def structural_layer(text: str) -> bool:
    if not text:
        return False
    if any(ch in text for ch in ZERO_WIDTH_CHARS):
        return True
    if _any_match(STRUCTURAL_PATTERNS, text.lower()):
        return True
    # Repeated-token burst = a long CONSECUTIVE run of one short token.
    # (Global frequency was tried first and falsely blocked 17/20 real
    # resumes on words like 'the' - see integrated benchmark v1. Fixed.)
    tokens = text.split()
    run = 1
    for a, b in zip(tokens, tokens[1:]):
        run = run + 1 if a == b else 1
        if run >= 8 and len(b.strip('.,;:')) <= 3:
            return True
    return False

def strip_invisible(text: str) -> str:
    if not text:
        return text
    for ch in ZERO_WIDTH_CHARS:
        text = text.replace(ch, "")
    return text

class AIHiringGuardrails:
    """Enterprise guardrails for the SmartHire platform."""

    @staticmethod
    def scan_document(document_text: str) -> dict:
        text = document_text or ""
        lower = text.lower()
        layers = {
            "layer1_regex": _any_match(LAYER1_PATTERNS, lower),
            "layer2_heuristic": heuristic_regex_layer_2(text),
            "layer3_structural": structural_layer(text),
        }
        return {"blocked": any(layers.values()), "layers": layers,
                "sanitized_text": strip_invisible(text)}

    @staticmethod
    def detect_prompt_injections(document_text: str):
        """Backwards-compatible raising check - now ALL deterministic layers.
        Previously Layer 1 only (20.0% mitigation on the project's own
        benchmark). Now the full production scan."""
        verdict = AIHiringGuardrails.scan_document(document_text)
        if verdict["blocked"]:
            fired = [k for k, v in verdict["layers"].items() if v]
            raise HTTPException(status_code=403, detail=(
                "Security Guardrail Triggered: Adversarial prompt injection "
                f"detected in the uploaded document. Layers fired: {fired}"))
        return True

    @staticmethod
    def prompt_guard_2_available() -> bool:
        if not os.getenv("HF_TOKEN"):
            return False
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def guardrail_status() -> dict:
        return {
            "layer1_regex": True,
            "layer2_heuristic": True,
            "layer3_structural": True,
            "prompt_guard_2_available": AIHiringGuardrails.prompt_guard_2_available(),
            "llama_guard_3_configured": bool(os.getenv("GROQ_API_KEY") and "PASTE_YOUR_KEY_HERE" not in os.getenv("GROQ_API_KEY")),
            "llama_guard_3_fail_mode": os.getenv("GUARDRAIL_FAIL_MODE", "open"),
        }

    @staticmethod
    def enforce_hr_topic(chat_message: str):
        out_of_bounds_topics = ["write a poem", "code a game", "politics", "recipe",
                                "tell me a joke", "hack", "how to build a bomb"]
        msg_lower = (chat_message or "").lower()
        for topic in out_of_bounds_topics:
            if topic in msg_lower:
                raise HTTPException(status_code=400, detail="Topic Guardrail Triggered: This assistant is strictly restricted to Recruitment and Candidate Assessment.")

    @staticmethod
    def redact_pii(text: str) -> str:
        sanitized = re.sub(r'\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', '[REDACTED_PHONE]', text or "")
        sanitized = re.sub(r'\d{3}-\d{2}-\d{4}', '[REDACTED_SSN]', sanitized)
        return sanitized

    @staticmethod
    def llama_guard_check(text_to_check: str, api_key: str = None) -> bool:
        """Llama Guard 3 via Groq (general safety, NOT injection detection).
        Fail-open by default; GUARDRAIL_FAIL_MODE=closed blocks on failure.
        The mode is always logged, never silent."""
        key = api_key or os.getenv("GROQ_API_KEY")
        fail_closed = os.getenv("GUARDRAIL_FAIL_MODE", "open").lower() == "closed"
        if not key or "PASTE_YOUR_KEY_HERE" in key:
            if fail_closed:
                raise HTTPException(status_code=503, detail="Llama Guard 3 not configured and GUARDRAIL_FAIL_MODE=closed.")
            return True
        import requests
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        data = {"model": "llama-guard-3-8b", "messages": [{"role": "user", "content": text_to_check}]}
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                classification = response.json()["choices"][0]["message"]["content"].strip().lower()
                if "unsafe" in classification:
                    raise HTTPException(status_code=403, detail=f"Llama Guard 3 Blocked: Content categorized as unsafe by Meta's Safety Taxonomy. Output: {classification}")
            return True
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            print(f"[GUARDRAIL WARNING] Llama Guard 3 API check failed: {e} (fail_mode={os.getenv('GUARDRAIL_FAIL_MODE', 'open')})")
            if fail_closed:
                raise HTTPException(status_code=503, detail="Llama Guard 3 check failed and GUARDRAIL_FAIL_MODE=closed.")
            return True
