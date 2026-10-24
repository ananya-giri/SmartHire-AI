# SmartHire AI — Final Research Report (v3, Gap-Closure Edition)

## Explainable, Secure and Neuro-Symbolic Recruitment Intelligence: Knowledge-Graph Skill Matching, Closed-Form Shapley Attribution and Production Prompt-Injection Defence — Integrated and Ablated

**Codebase:** github.com/ananya-giri/Smart-Hire (v3 fixed edition, delivered with this report)
**Date:** 21 October 2026
**Status:** All three research gaps addressed in code and re-evaluated end to end. Every number in this report was produced by a script committed in the repository and re-run for this edition.

## Abstract

SmartHire AI is a multi-agent recruitment system built around three components: a neuro-symbolic knowledge graph for skill matching, a closed-form Shapley attribution engine that explains the deterministic matching score, and a multi-layer prompt-injection guardrail. An independent audit of the previous version (v2) found that these components existed but did not add up to the claims made for them: the live pipeline mis-wired the graph and the explainer, the strongest guardrail layers existed only in the benchmark harness, and no experiment evaluated the components together. This v3 edition fixes those defects, wires the benchmarked components into the production request path, and adds the missing integrated evaluation. On 130 skill cases the knowledge graph scores 100% accuracy against 43.1% for exact keyword matching (labels for positive cases are graph-derived, so this measures implementation fidelity, not real-world validity). On 20 real resumes with proxy relevance labels, adding the graph raises screening F1 at the operational threshold from 0.333 to 0.966; TF-IDF ranks marginally better by AUC (1.000 vs 0.960), an honest counter-result we analyse rather than hide. Adding Shapley attribution changes no score and provides explanations for 100% of candidates, with perturbation faithfulness (AOPC comprehensiveness 27.29 vs sufficiency 3.14 on 14 eligible candidates) beyond the algebraic additivity identity. Adding the production guardrail changes no clean-resume score (0 of 20 real resumes blocked) while mitigating 76.7% of attack documents on the development security set and 56.7% on the held-out set at 0% false-positive rate. The contribution is a systems and evaluation contribution — an integrated, reproducible, honestly-bounded pipeline — not a new matching or attribution algorithm.

## 1. The three research gaps

Gap 1 — Limited integration of explainable skill matching. Keyword screening cannot represent skill relationships and cannot say why a candidate matches. SmartHire targets this with graph-based matching plus Shapley-style attribution over the matching score itself.

Gap 2 — Insufficiently evaluated security safeguards. LLM recruitment tools ingest untrusted documents, yet guardrails are rarely measured, and almost never verified to be the ones actually running in production. SmartHire targets this with layered deterministic guardrails, measured on attack and benign cases, against the production code path.

Gap 3 — Limited reproducible evaluation of integrated workflows. Components evaluated separately cannot demonstrate combined value. SmartHire targets this with an integrated ablation benchmark: six configurations of one pipeline, on the same data, end to end.

## 2. Related work and novelty positioning

Shapley values originate in cooperative game theory (Shapley, 1953) and were popularised for model explanation by SHAP (Lundberg and Lee, 2017); for a linearly separable scoring game the closed form used here is a direct consequence, not a new estimator. Graph-based skill matching builds on a long line of ontology-based recruitment and on public skill taxonomies such as ESCO, which covers orders of magnitude more skills than the 70-node graph used here. Prompt injection against LLM applications, including indirect injection through ingested documents, was characterised by Greshake et al. (2023) and Perez and Ribeiro (2022); dedicated classifiers such as Llama Guard (Inan et al., 2023) and Llama Prompt Guard address detection, and are used or evaluated here as components, not contributions. Shortest-path matching uses Dijkstra's algorithm (1959) unchanged, and explanation faithfulness is assessed with perturbation methods in the tradition of sanity-check and removal-based evaluation. The defensible novelty of SmartHire is therefore narrow and stated plainly: (i) a deterministic graph scorer whose output — not an LLM's text — is the decision score, explained exactly by construction; (ii) a production guardrail that is the same code the benchmarks measure, with an endpoint that reports what is active; and (iii) an integrated ablation protocol that prices each component's contribution and cost on common data. Claims beyond that are not made.

## 3. System and methods (as implemented in v3)

Knowledge graph. An undirected NetworkX graph with 70 skill nodes and 74 edges; integer weights (1 for ordinary relations, 2 for looser domain links, 3–5 for cross-domain bridges). A curated alias table (about 55 entries) resolves synonyms before graph lookup. A requirement is approved when the Dijkstra shortest-path distance to the candidate's closest extracted skill is at most 2, flagged at 3, and a gap beyond. The candidate score is the percentage of job requirements approved — binary per requirement, deterministic, and identical on re-runs.

Shapley attribution. For requirement j with approval state z in {0,1} and cohort baseline rate p, the attribution is phi = (100/M)(z − p), where M is the number of requirements and baselines are the observed approval rates over the 20-resume Kaggle cohort (provenance committed in data/shap_baseline_provenance.json). At skill level, each requirement's 100/M points are split equally among the candidate skills that satisfy it; skills satisfying nothing receive exactly zero (null-player property). Sum of attributions equals score minus expected score exactly — an algebraic identity, treated in this report as a correctness check, never as empirical evidence. Empirical faithfulness is measured separately by AOPC perturbation: removing top-attributed skills must damage the score more than removing null skills.

Guardrail (production). security_guardrails.scan_document, called by the API on resume and job description and again inside the pipeline before scoring. Layer 1: direct-imperative regex (instruction overrides, system-prompt references, score coercion, instruction-token spoofing). Layer 2: semantic-intent regex (paraphrased subversion, role-play framing, tool-exfiltration phrasing, coercion actions). Layer 3: structural checks (invisible unicode, directive delimiter blocks, consecutive-token bursts). PII redaction follows. Llama Guard 3 is a separate general-safety layer with an explicit, logged fail mode and is never credited in injection metrics. Prompt-Guard-2 is a benchmark comparator only. GET /guardrail/status reports the active layers of a deployment; POST /score returns the deterministic score and attributions for one candidate with no LLM involved.

Agents. Eleven CrewAI agents (ten specialists plus an executive manager) in a four-tier DAG narrate, source, audit and synthesise around the deterministic core. The screening agent is instructed to report the deterministic score unchanged, and the delivered dossier always appends the verified deterministic section verbatim, so the auditable result cannot be lost or altered by LLM synthesis.

## 4. What the v2 audit found, and what v3 fixed

| # | v2 defect | v3 fix |
|---|---|---|
| 1 | Pipeline passed resume prose to a function expecting a skill list, for 10 hardcoded keywords only | Requirements and candidate skills are extracted from JD and resume text; the graph is evaluated per extracted requirement |
| 2 | Job-description string passed as the requirements list to the Shapley report (iterated over characters) | Correct list arguments everywhere, plus a normaliser in the explainer that parses any string argument as JD text; regression test added |
| 3 | Decision score generated by an LLM; the explainer explained a different, unused score | Deterministic Graph_ATS score is authoritative, reported unchanged by agents and appended verbatim to the dossier |
| 4 | Fabricated fallback strings silently replaced failures (hardcoded percentages, "verified" audits) | Removed; failures surface as explicit UNAVAILABLE markers with the underlying error |
| 5 | Guardrail Layers 2 and Prompt-Guard-2 existed only in the benchmark harness; production ran Layer 1 alone (20.0% mitigation) | All deterministic layers moved into the production module; the benchmark re-exports the production implementation; API and pipeline both call it |
| 6 | Report described a directed, typed graph (0.05/0.15/0.25) and security/rendering details absent from code; security N given as 60; a SHAP latency table had no backing data | Documentation rewritten to the implementation; N corrected to 55; unbacked table removed; measured values only |
| 7 | No ablation, no end-to-end experiment, no held-out generator, unpinned benchmark dependencies | run_integrated_benchmarks.py, generate_heldout_cases.py and requirements-benchmark.txt added; 9 active-application tests added |

Two further defects were found by the new integrated benchmark while fixing the above, and are recorded because they are exactly what integrated evaluation is for. First, a structural heuristic based on global token frequency falsely blocked 17 of 20 real resumes; it was replaced with consecutive-run detection, after which 0 of 20 are blocked. Second, generated held-out paraphrases showed that Layer 2 depended on exact word order and was defeated by a single article; the patterns were generalised linguistically and every set was re-run. As a consequence the generated v3 security set is treated as a regression set, not a pristine held-out set.

## 5. Integrated evaluation

Design. Six configurations of one pipeline — A exact keyword, B alias-normalised, C TF-IDF cosine, D knowledge graph, E graph plus Shapley, F full (graph plus Shapley plus production guardrail) — evaluated on the same data by run_integrated_benchmarks.py: 130 skill cases (40 dev, 50 held-out scored, 40 generated v3), 20 real Kaggle resumes against the ten Senior Full-Stack requirements (proxy relevance label: category is not Accountant; decision threshold 50% declared in advance), explanation quality over those resumes, and the production guardrail over 141 security cases plus the 20 real resumes. Full per-case results are committed (data/integrated_per_case_results.json).

### 5.1 Skill-case accuracy

| Scorer | Dev (40) | Held-out (50) | Generated v3 (40) | All (130) |
|---|---|---|---|---|
| Exact keyword | 45.0% | 40.0% | 45.0% | 43.1% |
| Alias-normalised | 70.0% | 60.0% | 62.5% | 63.8% |
| Knowledge graph | 100.0% | 100.0% | 100.0% | 100.0% |

Interpretation is deliberately restrained: positive cases were constructed from graph neighbourhoods and the alias table, so a correct implementation must score near-perfectly. This table verifies that the implementation does what the method specifies and quantifies how much of that behaviour keyword and alias matching miss. It is not evidence about real-world hiring validity.

### 5.2 Resume screening and ablation (N=20, proxy labels, threshold 50%)

| Config | F1 | Accuracy | Recall | FRR | AUC | Mean scorer latency |
|---|---|---|---|---|---|---|
| A Exact | 0.333 | 40.0% | 20.0% | 80.0% | 0.933 | 0.02 ms |
| B Alias | 0.333 | 40.0% | 20.0% | 80.0% | 0.933 | 0.03 ms |
| C TF-IDF | 0.500 | 50.0% | 33.3% | 66.7% | 1.000 | – |
| D KG | 0.966 | 95.0% | 93.3% | 6.7% | 0.960 | 3.73 ms |
| E KG + SHAP | 0.966 | 95.0% | 93.3% | 6.7% | 0.960 | 3.69 ms |
| F Full | 0.966 | 95.0% | 93.3% | 6.7% | 0.960 | 4.64 ms |

Ablation readings. Graph versus exact: F1 +0.632, false rejections 80.0% to 6.7% — the graph is where the screening value is. Alias adds nothing on this corpus because extracted tokens are already canonical; its value appears in the skill cases instead. TF-IDF out-ranks the graph by AUC (1.000 vs 0.960) because its scores are continuous while binary graph approvals saturate and tie, yet at the operational threshold its F1 is roughly half the graph's — ranking quality and decision quality are different properties, and a deployment choosing a ranker should weigh this result seriously. Shapley costs no accuracy (delta F1 0.000) and buys complete explanation coverage. The guardrail costs no clean-data accuracy (delta F1 0.000; 0 of 20 real resumes blocked) — its price is paid only in the security table below. Full explanation generation (skill-level attribution with perturbation audit) takes roughly 0.5–0.7 s per candidate in the reference environment, versus about 4 ms for scoring plus guardrail; explanations are therefore best generated on demand or asynchronously.

### 5.3 Explanation quality (configs E and F)

Coverage 100% of candidates. Maximum additivity error 0.0 (identity check). AOPC perturbation faithfulness on the 14 eligible candidates (at least two extracted skills; six degenerate profiles excluded and identified in the artifacts): comprehensiveness 27.29, sufficiency 3.14, pass rate 100%. A worked, verifiable example is available without any LLM via POST /score; the regression test in the repository shows a resume phrased with a framework and an alias spelling being approved for the corresponding foundation and canonical database requirements, with the attributions summing exactly to the score shift.

### 5.4 Security: the production guardrail, measured

| Set | Attacks | Benign | Mitigation | FPR |
|---|---|---|---|---|
| Dev | 30 | 25 | 76.7% | 0.0% |
| Held-out | 30 | 25 | 56.7% | 0.0% |
| Generated v3 (regression set) | 15 | 16 | 100.0% | 0.0% |
| Real resumes | – | 20 | – | 0 blocked |

Reference points: the v2 production path (Layer 1 alone) mitigates 20.0% on the dev set; Prompt-Guard-2 alone mitigated 26.7% (dev) and 43.3% (held-out) in the pre-fix artifacts. The held-out figure of 56.7% is the honest headline for unseen attacks — better than any single layer previously measured, and far from complete. Unguarded configurations admit every attack document to scoring by construction, so the guardrail's integrated benefit is the difference between 100% admission and the rates above, at zero measured clean-data cost.

## 6. Discussion

The three gaps close unevenly, and the report grades them unevenly. Gap 1 is addressed at system level: matching and explanation are integrated, the explained score is the decision score, and explanations are verifiable per candidate over an API. What remains unproven is validity — graph-derived labels and proxy relevance cannot show that approved candidates are actually competent, and the TF-IDF AUC result warns against assuming graph superiority in every respect. Gap 2 is addressed for the deterministic production guardrail, with the active-application verification the gap demanded; broader attack coverage (encoded, multilingual, PDF-layer and indirect injections through retrieved content) and a measured Prompt-Guard-2 production integration remain open, and 56.7% held-out mitigation should be read as a floor to improve, not a defence guarantee. Gap 3 is addressed by construction: the integrated ablation now exists, is reproducible from committed scripts and artifacts, and has already earned its keep twice by exposing defects that component benchmarks missed. Its datasets are small and partly synthetic, so it is a baseline protocol for larger studies rather than a definitive evaluation.

## 7. Limitations and threats to validity

1. No human ground truth: no hiring outcomes, recruiter judgements or competence labels exist in any dataset used; the resume relevance label is a category proxy.
2. Skill-case labels are graph-derived for positives; perfect graph accuracy is expected by construction.
3. Small samples throughout (20 resumes; 30 attacks per security set); no confidence intervals are reported and percentage differences of a few points are within noise.
4. The generated v3 security set was used during development of the generalised Layer-2 patterns and is a regression set; only the pre-existing held-out set approximates unseen attacks, and it shares authorship with the dev set.
5. The graph covers 70 technology skills for software roles only; extraction is ontology-bound, so skills outside the graph are invisible to the deterministic scorer.
6. Security evaluation measures document-level detection, not downstream attack success against the LLM agents; a payload that passes the guardrail may still fail to change an outcome, and one that is blocked at document level says nothing about injections arriving through the vault, web or GitHub channels, which are not yet guarded or measured.
7. Fairness, video-integrity, GitHub-audit and market features have no labelled evaluation in this edition and no accuracy is claimed for them.
8. Latencies are single-environment measurements and vary by machine.

## 8. Reproducibility

From the repository root, with the pinned deterministic stack (requirements-benchmark.txt): generate the held-out v3 cases with python generate_heldout_cases.py (seed 42); run the integrated benchmark with python run_integrated_benchmarks.py; run the Shapley pipeline benchmark with python run_shap_pipeline_benchmarks.py; verify the active application with python -m unittest test_active_guardrail_and_pipeline.py (9 tests, all passing in this edition); verify historical artifact arithmetic with python test_metric_reproducibility.py (passing). Key artifacts: data/integrated_benchmark_metrics.json, data/integrated_per_case_results.json, data/heldout_v3_cases.json, data/shap_pipeline_metrics_v2.json, data/security_benchmark_v2.json, data/heldout_aggregate_metrics.json. The full change list is in FIXES_V3.md.

## 9. Conclusion

After the v3 fixes, SmartHire's code, documentation and evidence tell the same story. Explainable matching is genuinely integrated: a deterministic graph score decides, and exact Shapley attributions explain that score for every candidate. Security is genuinely production-grade in structure: the measured guardrail is the running guardrail, verified by tests against the application path, with an honest held-out mitigation of 56.7% at zero false positives. And the integrated workflow is genuinely evaluated: one ablation prices the graph (+0.632 F1), the explanations (zero score cost, full coverage, perturbation-faithful) and the guardrail (zero clean-data cost, majority attack mitigation) on common data. The remaining work is validity at scale — human-labelled candidates, independent attack corpora, and a larger skill ontology — which this edition's protocol is designed to support.

## References

Shapley, L. S. (1953). A value for n-person games. In Contributions to the Theory of Games, Vol. II. Princeton University Press.
Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. Numerische Mathematik, 1, 269–271.
Lundberg, S. M., and Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30 (NeurIPS).
Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. Proc. ACM SIGKDD.
Greshake, K., et al. (2023). Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. arXiv:2302.12173.
Perez, F., and Ribeiro, I. (2022). Ignore previous prompt: Attack techniques for language models. arXiv:2211.09527.
Inan, H., et al. (2023). Llama Guard: LLM-based input-output safeguard for human-AI conversations. arXiv:2312.06674.
European Commission (2022). ESCO — European Skills, Competences, Qualifications and Occupations, v1.2.
