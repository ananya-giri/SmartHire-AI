# SmartHire AI — System Architecture (corrected, v3)

SmartHire AI is a multi-agent recruitment platform: a FastAPI backend
(`api.py`), a CrewAI orchestration layer (`main.py`, 11 agents in a 4-tier
DAG), a React/Vite frontend (`frontend/`), and a Streamlit demo
(`ui/app.py`) that exercises the deterministic scoring stack directly.

> This file previously described a different system ("TalentSpark AI",
> 4 agents, Streamlit-only). It has been rewritten to match the code.

## Request flow (production)

```
Upload (resume + JD [+ video])
  -> api.py
       1. PDF/text extraction
       2. PRODUCTION GUARDRAIL scan_document() on resume and JD
          (Layer 1 regex + Layer 2 heuristic + Layer 3 structural;
           blocked -> error, nothing is scored)
       3. Llama Guard 3 safety check (external, explicit fail mode)
       4. PII redaction
       5. Resume vault (ChromaDB)
  -> main.run_recruitment_flow()  (async job or synchronous)
       0. Guardrail scan again in-pipeline (defence in depth; quarantine)
       1. Deterministic pre-computation (AUTHORITATIVE):
            requirements  = extract_job_requirements(JD)
            skills        = extract_candidate_skill_tokens(resume)
            Graph_ATS score = calculate_pipeline_match_score(skills, reqs)
            KG details per requirement (evaluate_skill_match, Dijkstra)
            Shapley report (generate_pipeline_shap_report)
       2. Tier 1 agents: market, CV integrity, sourcing
       3. Tier 2 agent: screening (must report the Graph_ATS score unchanged)
       4. Tier 3 agents: assessment/GitHub, fairness, explainability,
          engagement, analytics, onboarding
       5. Tier 4 agent: executive dossier synthesis
       6. The verified deterministic section is appended verbatim to
          whatever the LLM produced.
```

## Deterministic, LLM-free endpoints

- `POST /score` — Graph_ATS score, per-requirement match details and
  Shapley attributions for one candidate. No LLM is invoked.
- `GET /guardrail/status` — which guardrail layers are active.
- `GET /health`, `POST /analyze`, `GET /analyze/status/{id}`,
  `POST/GET /vault`, `POST /chat`.

## Component map

| Concern | Implementation | Used by |
|---|---|---|
| Skill graph | `tools/knowledge_graph_tool.py` — undirected NetworkX graph, 70 nodes / 74 edges, integer weights, Dijkstra, alias table | pipeline, `/score`, benchmarks |
| Deterministic scorer + Shapley | `tools/shap_scoring_explainer.py` | pipeline, `/score`, Streamlit demo, benchmarks |
| Guardrails | `security_guardrails.py` (single source of truth) | `api.py`, `main.py`, `benchmark_suite.py` (re-export) |
| Agents | `agents/recruitment_agents.py` — 11 agent factories | `main.py` |
| Benchmarks | `benchmark_suite.py`, `run_integrated_benchmarks.py`, `run_shap_pipeline_benchmarks.py`, `run_security_benchmark_v2.py`, `generate_heldout_cases.py` | research evaluation |

## Known boundaries

- The Shapley explainer explains the deterministic Graph_ATS score only;
  LLM narrative text is not (and cannot be) explained by it.
- Video integrity, GitHub auditing and market prediction are LLM/tool
  features without labelled evaluation datasets; no accuracy is claimed
  for them.
- The LLM (Gemini via `config.py`, or Groq if configured) narrates and
  synthesises; it does not set the match score.
