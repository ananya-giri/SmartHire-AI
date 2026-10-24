# SmartHire v3 — Gap-Closure Fixes (2026-10-21)

This changelog records every change made to close the three research gaps,
including two defects the new integrated benchmark found in the fix itself.

## Gap 1 — Explainable skill matching
1. `main.py` wiring (critical): requirements are now extracted from the JD
   (`extract_job_requirements`) and candidate skills from resume prose
   (`extract_candidate_skill_tokens`); `evaluate_skill_match` receives a
   skill list, not raw prose. Previously it received the whole resume text
   for 10 hardcoded keywords only.
2. `main.py` SHAP call (critical): `generate_pipeline_shap_report` now
   receives (candidate_skills_list, required_skills_list). Previously the
   JD string was passed as `required_skills` and iterated character by
   character.
3. `tools/shap_scoring_explainer.normalize_required_skills` (new): a string
   argument is always parsed as JD text, so this bug class cannot recur.
   Covered by a regression test.
4. The deterministic Graph_ATS score is now the authoritative match score:
   the screening prompt instructs the LLM to report it unchanged, and the
   final dossier always appends the verified deterministic section
   (score, requirements, skills, KG details, Shapley report) verbatim.
5. All fabricated fallback strings removed from `main.py` (hardcoded SHAP
   percentages, "verified competencies", "GitHub verified"). Failures now
   surface as explicit `UNAVAILABLE: <error>` markers.
6. `api.py` adds `POST /score`: deterministic score + per-requirement
   Shapley attributions for a real candidate, with no LLM involved —
   real-candidate explanations are now directly verifiable over the API.
7. `tools/knowledge_graph_tool`: module docstring corrected to the actual
   implementation (undirected, untyped, integer weights, binary approval)
   and `get_graph_stats()` added (70 nodes, 74 edges, verified by test).
   The deterministic tools now import without the CrewAI stack.

## Gap 2 — Security safeguards
1. `security_guardrails.py` rewritten as the single source of truth:
   Layer 1 (direct regex, extended with instruction-token spoofing and
   score-coercion patterns), Layer 2 (semantic heuristic — moved from
   `benchmark_suite.py`, which now re-exports it), Layer 3 (structural:
   zero-width characters, delimiter blocks, consecutive-token bursts).
2. `scan_document()` is the production entry point; `detect_prompt_injections`
   now runs all layers (previously Layer 1 only = 20.0% mitigation).
   `api.py /analyze` scans resume AND JD with it; `main.py` scans again
   in-pipeline and quarantines before scoring.
3. `GET /guardrail/status` reports which layers are active in a deployment.
   Llama Guard 3 is documented as a general safety layer (not credited in
   injection metrics) with an explicit, logged fail mode
   (`GUARDRAIL_FAIL_MODE=open|closed`).
4. `test_active_guardrail_and_pipeline.py` (new, 9 tests) verifies the
   ACTIVE guardrail, not a benchmark copy.

### Defects found by the integrated benchmark during this fix
- Structural v1 used global token frequency and falsely blocked 17/20 real
  resumes (common words repeat). Replaced with consecutive-run detection;
  real-resume block rate is now 0/20.
- Held-out v3 templates exposed that Layer 2 required the exact word order
  "...aside earlier..." — a single article ("the earlier") defeated it.
  Layer 2 was generalised (optional articles/pronouns, role-play wording,
  generic tool-execution and score-coercion phrases) and ALL sets were
  re-run. Consequence for interpretation: v3 is now a regression set, not
  a pristine held-out set; the pre-existing held-out set (56.67%) is the
  least-touched mitigation estimate.

## Gap 3 — Integrated, reproducible evaluation
1. `run_integrated_benchmarks.py` (new): six pipeline configurations
   (exact / alias / TF-IDF / KG / KG+SHAP / full) on the same data —
   130 skill cases (dev + held-out + v3), 20 real resumes (proxy labels,
   disclosed), explanation quality (additivity, AOPC), and the production
   guardrail on 141 security cases + the real resumes. Outputs committed
   as `data/integrated_benchmark_metrics.json` and per-case results.
2. `generate_heldout_cases.py` (new): the previously missing held-out
   generator (seed 42), with label basis and limits in its provenance.
3. `requirements-benchmark.txt` (new): pinned deterministic stack.
4. README and RESEARCH_REPORT corrected (see below); ARCHITECTURE.md
   rewritten to describe the actual system.

## Corrected claims
- Security benchmark is N=55 (30 attacks + 25 benign), not 60/30+30.
- The 76.7% figure is the production combined regex guardrail on the dev
  set — it is no longer attributed to Prompt-Guard-2 (PG2 alone: 26.7%
  dev; combined-with-PG2 held-out: 46.7% in the pre-fix artifacts).
- The knowledge graph is undirected with integer weights 1–5 and binary
  per-requirement approval — not directed typed edges at 0.05/0.15/0.25
  with a continuous distance-decayed score.
- The former RESEARCH_REPORT §4.3 SHAP latency/feature table had no
  backing code or data and has been removed; measured latencies are in
  the integrated benchmark output.
- Agent count is 11 (10 specialists + executive manager).
