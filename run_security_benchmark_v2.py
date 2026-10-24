"""
run_security_benchmark_v2.py — Enhanced Security Benchmark Runner
=================================================================
Runs the full 55-case security guardrail benchmark and writes two versioned
output artefacts:

  data/security_benchmark_v2.json       — aggregate summary (unchanged schema)
  data/security_per_case_v2.json        — per-case records (unchanged schema)

Additionally writes a new transparency artefact:

  data/llama_guard2_per_case_scores.json — per-case raw inference metadata
    For every case:
      sample_id, family, is_attack, predicted_label, inject_prob, safe_prob,
      above_threshold, model_id, threshold, inference_status, inference_error,
      model_loaded, package_versions

This script is designed for full reproducibility verification.
If HF_TOKEN is not set, LlamaPromptGuard2 predictions will all be PASSED
(model unavailable fallback) AND a clear warning is emitted.

Usage:
    python run_security_benchmark_v2.py [--preserve-existing]

Options:
    --preserve-existing   Do NOT overwrite security_benchmark_v2.json and
                          security_per_case_v2.json (audit preservation mode).
                          The llama_guard2_per_case_scores.json is always written.
"""

import os
import sys
import json
import time
import re
import platform
import importlib

from dotenv import load_dotenv

# ── resolve project root ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── load .env file (HF_TOKEN, GROQ_API_KEY, etc.) ────────────────────────────
load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)
sys.path.insert(0, BASE_DIR)

import benchmark_suite as bs

# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "data")
BENCHMARK_PATH   = os.path.join(DATA_DIR, "security_benchmark_v2.json")
PER_CASE_PATH    = os.path.join(DATA_DIR, "security_per_case_v2.json")
SCORES_PATH      = os.path.join(DATA_DIR, "llama_guard2_per_case_scores.json")

PRESERVE_EXISTING = "--preserve-existing" in sys.argv


# ── package version collector ─────────────────────────────────────────────────

def collect_package_versions():
    pkgs = {
        "torch": None, "transformers": None, "huggingface_hub": None,
        "numpy": None, "python": platform.python_version(),
    }
    for pkg in ("torch", "transformers", "huggingface_hub", "numpy"):
        try:
            mod = importlib.import_module(pkg)
            pkgs[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pkgs[pkg] = "not_installed"
    return pkgs


# ── per-case LlamaGuard2 inference with raw score capture ────────────────────

def llama_guard2_predict_with_scores(text: str, tok, mdl, threshold: float):
    """
    Run one forward pass and return a detailed result dict:
      {
        inject_prob   : float | None,
        safe_prob     : float | None,
        predicted_label: 'BLOCKED' | 'PASSED' | 'SKIPPED',
        above_threshold: bool | None,
        inference_status: 'ok' | 'model_unavailable' | 'error',
        inference_error : str | None,
      }
    Never raises; errors are captured in inference_error.
    """
    if tok is None or mdl is None:
        return {
            "inject_prob":    None,
            "safe_prob":      None,
            "predicted_label": "SKIPPED",
            "above_threshold": None,
            "inference_status": "model_unavailable",
            "inference_error":  "HF_TOKEN not set or model failed to load",
        }
    try:
        import torch
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = mdl(**inputs).logits
        probs       = torch.softmax(logits, dim=-1)
        safe_prob   = round(probs[0][0].item(), 6)
        inject_prob = round(probs[0][1].item(), 6)
        above_threshold = inject_prob > threshold
        return {
            "inject_prob":    inject_prob,
            "safe_prob":      safe_prob,
            "predicted_label": "BLOCKED" if above_threshold else "PASSED",
            "above_threshold": above_threshold,
            "inference_status": "ok",
            "inference_error":  None,
        }
    except Exception as exc:
        return {
            "inject_prob":    None,
            "safe_prob":      None,
            "predicted_label": "PASSED",   # safe fallback; never count as TP
            "above_threshold": None,
            "inference_status": "error",
            "inference_error":  "%s: %s" % (type(exc).__name__, exc),
        }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Security Benchmark v2 — Enhanced Runner")
    print("Timestamp:", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("Python:", platform.python_version(), "on", platform.system(), platform.release())
    print("Preserve existing artefacts:", PRESERVE_EXISTING)
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)
    pkg_versions = collect_package_versions()
    print("Package versions:", pkg_versions)
    print()

    # ── 1. Run the benchmark (exactly as before) ─────────────────────────────
    t0 = time.time()
    summary, per_case = bs.run_security_benchmark()
    elapsed = time.time() - t0

    # ── 2. Write aggregate artefacts ─────────────────────────────────────────
    if PRESERVE_EXISTING:
        print("\n[INFO] --preserve-existing: skipping overwrite of security_benchmark_v2.json "
              "and security_per_case_v2.json (audit evidence preserved).")
    else:
        with open(BENCHMARK_PATH, "w", encoding="utf-8") as fh:
            json.dump({
                "benchmark":   "Security_Guardrails_v2",
                "elapsed_sec": round(elapsed, 2),
                "summary":     summary,
            }, fh, indent=2)
        print("\n[OK] Wrote:", BENCHMARK_PATH)

        with open(PER_CASE_PATH, "w", encoding="utf-8") as fh:
            json.dump(per_case, fh, indent=2)
        print("[OK] Wrote:", PER_CASE_PATH, "(%d records)" % len(per_case))

    # ── 3. Generate per-case raw LlamaGuard2 scores ──────────────────────────
    print("\n[INFO] Generating per-case LlamaGuard2 raw score artefact...")
    model_id  = bs._PG2_MODEL_ID
    threshold = bs._PG2_THRESHOLD

    # Load (or reuse cached) model
    tok, mdl = bs._load_llama_prompt_guard_2()
    pg2_available = (tok is not None)
    if not pg2_available:
        print("[WARN] LlamaPromptGuard2 model NOT loaded. "
              "Raw scores will be None and all predictions will be SKIPPED.\n"
              "       Set HF_TOKEN=<your_hf_token> and re-run to obtain genuine ML scores.")

    # Rebuild full sample list (same order as benchmark_suite.run_security_benchmark)
    all_samples = []
    for payload in bs.FROZEN_SECURITY_PAYLOADS:
        all_samples.append({
            "id": payload["id"], "family": payload["family"],
            "is_attack": True, "text": payload["text"],
        })
    for idx, text in enumerate(bs.BENIGN_CONTROLS):
        all_samples.append({
            "id": "BEN-%02d" % (idx + 1), "family": "Benign_Control",
            "is_attack": False, "text": text,
        })

    per_case_scores = []
    for s in all_samples:
        result = llama_guard2_predict_with_scores(s["text"], tok, mdl, threshold)
        per_case_scores.append({
            "sample_id":       s["id"],
            "family":          s["family"],
            "is_attack":       s["is_attack"],
            "text_snippet":    s["text"][:80] + ("..." if len(s["text"]) > 80 else ""),
            "model_id":        model_id,
            "threshold":       threshold,
            "model_loaded":    pg2_available,
            "inject_prob":     result["inject_prob"],
            "safe_prob":       result["safe_prob"],
            "predicted_label": result["predicted_label"],
            "above_threshold": result["above_threshold"],
            "inference_status": result["inference_status"],
            "inference_error": result["inference_error"],
            "package_versions": pkg_versions,
        })

    scores_artefact = {
        "artefact_type":  "llama_guard2_per_case_scores",
        "generated_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id":       model_id,
        "threshold":      threshold,
        "model_loaded":   pg2_available,
        "inference_type": "genuine_forward_pass" if pg2_available else "model_unavailable",
        "package_versions": pkg_versions,
        "total_cases":    len(per_case_scores),
        "note": (
            "Raw inject_prob and safe_prob from meta-llama/Llama-Prompt-Guard-2-22M "
            "forward pass (softmax over logits). predicted_label = BLOCKED if "
            "inject_prob > threshold, else PASSED. "
            "SKIPPED means the model was not loaded (HF_TOKEN absent or load failure)."
        ),
        "per_case": per_case_scores,
    }

    with open(SCORES_PATH, "w", encoding="utf-8") as fh:
        json.dump(scores_artefact, fh, indent=2)
    print("[OK] Wrote:", SCORES_PATH, "(%d records)" % len(per_case_scores))

    # ── 4. Print summary ─────────────────────────────────────────────────────
    print("\n=== OVERALL RESULTS ===")
    for comp, m in summary["overall"].items():
        print("  %-25s  Mitigation=%.1f%%  Prec=%.1f%%  F1=%.4f  FPR=%.1f%%" % (
            comp,
            m["recall_mitigation_pct"],
            m["precision_pct"],
            m["f1_score"],
            m["false_positive_rate_pct"],
        ))

    print("\n=== LLM GUARD INTEGRATION STATUS ===")
    for k, v in summary.get("llm_guard_integration_status", {}).items():
        print("  %s: %s" % (k, v))

    # Skipped-predictions warning
    skipped = [s["sample_id"] for s in per_case_scores if s["predicted_label"] == "SKIPPED"]
    if skipped:
        print("\n[WARN] %d LlamaPromptGuard2 predictions were SKIPPED (model not loaded)." % len(skipped))
        print("       Run with HF_TOKEN set to obtain genuine ML predictions.")
    else:
        print("\n[INFO] All %d LlamaPromptGuard2 predictions generated via genuine model inference." % len(per_case_scores))

    print("\n[INFO] Elapsed: %.2fs" % elapsed)
    print("=" * 70)


if __name__ == "__main__":
    main()
