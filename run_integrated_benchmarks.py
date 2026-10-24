"""
Integrated end-to-end + ablation benchmark (Gap-3 fix).

Before this script, every SmartHire component was benchmarked separately,
on different datasets, and there was no ablation: nothing measured what
each component contributes, or whether they work together.

This runner evaluates SIX configurations of the SAME pipeline on the SAME
data, end to end (guardrail -> matching -> explanation):

  A_exact        Exact keyword scorer, no guardrail, no explanation
  B_alias        Alias-normalised scorer, no guardrail, no explanation
  C_tfidf        TF-IDF cosine scorer (meaningful non-graph baseline),
                 no guardrail, no explanation
  D_kg           Knowledge-graph scorer (Graph_ATS), no guardrail
  E_kg_shap      Graph_ATS + Shapley attribution
  F_full         Graph_ATS + Shapley + PRODUCTION guardrail
                 (security_guardrails.scan_document - the code api.py runs)

Datasets (all committed in the repo):
  - Skill cases: data/per_case_results.json (dev, 40) +
                 data/heldout_per_case_results.json (held-out, 50 scored) +
                 data/heldout_v3_cases.json (generated, see generator)
  - Resumes:     data/extracted_kaggle_resumes.json (N=20, real resumes)
                 vs the Senior Full-Stack requirements (10 skills).
                 Proxy relevance label: category != "Accountant".
                 This is a PROXY (no human hiring ground truth exists in
                 the corpus) and is disclosed as such in every output.
  - Security:    dev (55) + held-out (55) + v3 generated cases, run through
                 the production guardrail; unguarded configurations admit
                 every attack document to scoring by construction.

Metrics: per-config accuracy/precision/recall/F1 at the pre-declared 50%
decision threshold (binary scorers), ranking AUC for all scorers,
explanation additivity + AOPC faithfulness (E, F), guardrail mitigation /
false-positive rate per security set (F), mean latency.

Outputs: data/integrated_benchmark_metrics.json
         data/integrated_per_case_results.json
Usage:   python run_integrated_benchmarks.py
Requires: networkx, numpy (scikit-learn only for C_tfidf; skipped if absent)
"""
import json
import os
import sys
import time
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
DATA = os.path.join(BASE_DIR, "data")

from tools.knowledge_graph_tool import evaluate_skill_match
from tools.shap_scoring_explainer import (
    calculate_pipeline_match_score, exact_keyword_match, alias_normalized_match,
    extract_candidate_skill_tokens, extract_job_requirements,
    compute_requirement_shap_values, compute_cohort_empirical_baselines,
    compute_skill_attribution_shapley, evaluate_feature_perturbation_faithfulness,
    DEFAULT_JD_REQUIREMENTS,
)
from security_guardrails import AIHiringGuardrails

DECISION_THRESHOLD = 50.0

from tools.knowledge_graph_tool import SKILL_ALIASES as _ALIASES

def token_exact(required_skill, cand_skills):
    return required_skill.strip().lower() in [t.strip().lower() for t in cand_skills]

def _norm(s):
    sl = s.strip().lower()
    sc = sl.replace("-", " ").replace("_", " ")
    if sl in _ALIASES: return _ALIASES[sl].lower()
    if sc in _ALIASES: return _ALIASES[sc].lower()
    return sl

def token_alias(required_skill, cand_skills):
    rn = _norm(required_skill)
    return any(_norm(t) == rn for t in cand_skills)
TFIDF_THRESHOLD = 10.0  # cosine * 100


def auc(scores_pos, scores_neg):
    if not scores_pos or not scores_neg:
        return None
    wins = 0.0
    for p in scores_pos:
        for n in scores_neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return round(wins / (len(scores_pos) * len(scores_neg)), 4)


def confusion(scores, labels, threshold):
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        pred = s >= threshold
        if pred and y: tp += 1
        elif pred and not y: fp += 1
        elif not pred and not y: tn += 1
        else: fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, tp + fp + tn + fn)
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy_pct": round(acc * 100, 2), "precision_pct": round(prec * 100, 2),
            "recall_pct": round(rec * 100, 2), "f1_score": round(f1, 4),
            "false_rejection_rate_pct": round(100 * fn / (tp + fn), 2) if tp + fn else 0.0}


def load_skill_cases():
    cases = []
    dev = json.load(open(os.path.join(DATA, "per_case_results.json")))
    for c in dev:
        if c.get("experiment") == "Skill_Matching_KG":
            cases.append({"set": "dev", "required_skill": c["required_skill"],
                          "candidate_skills": [t.strip() for t in c["candidate_skills"].split(",")],
                          "expected_label": c["expected_label"]})
    held = json.load(open(os.path.join(DATA, "heldout_per_case_results.json")))
    for c in held:
        if c.get("experiment") == "HeldOut_Skill_Matching" and c.get("expected_label") in ("APPROVED", "GAP"):
            cases.append({"set": "heldout", "required_skill": c["required_skill"],
                          "candidate_skills": [t.strip() for t in c["candidate_skills"].split(",")],
                          "expected_label": c["expected_label"]})
    v3_path = os.path.join(DATA, "heldout_v3_cases.json")
    if os.path.exists(v3_path):
        v3 = json.load(open(v3_path))
        for c in v3["cases"]:
            if c["kind"] == "skill":
                cases.append({"set": "heldout_v3", "required_skill": c["required_skill"],
                              "candidate_skills": c["candidate_skills"],
                              "expected_label": c["expected_label"]})
    return cases


def load_security_sets():
    sets = {}
    dev = json.load(open(os.path.join(DATA, "per_case_results.json")))
    sets["dev"] = [{"text": c["text"], "is_attack": bool(c["is_attack"])}
                   for c in dev if c.get("experiment") == "Security_Guardrails"]
    held = json.load(open(os.path.join(DATA, "heldout_per_case_results.json")))
    sets["heldout"] = [{"text": c["text"], "is_attack": str(c["is_attack"]) == "True"}
                       for c in held if c.get("experiment") == "HeldOut_Security_Guardrails"]
    v3_path = os.path.join(DATA, "heldout_v3_cases.json")
    if os.path.exists(v3_path):
        v3 = json.load(open(v3_path))
        sets["heldout_v3"] = [{"text": c["text"], "is_attack": c["is_attack"]}
                              for c in v3["cases"] if c["kind"] == "security"]
    return sets


def main():
    t_start = time.time()
    reqs = list(DEFAULT_JD_REQUIREMENTS)
    resumes = json.load(open(os.path.join(DATA, "extracted_kaggle_resumes.json")))
    jd_text = open(os.path.join(DATA, "job_description.md")).read() if os.path.exists(os.path.join(DATA, "job_description.md")) else ", ".join(reqs)

    # ---------- resume scoring per configuration ----------
    labels = [r.get("category") != "Accountant" for r in resumes]
    texts = [r.get("text", "") for r in resumes]
    skills_per = [extract_candidate_skill_tokens(t) for t in texts]

    def scorer_exact(skills):
        return 100.0 * sum(1 for r in reqs if token_exact(r, skills)) / len(reqs)

    def scorer_alias(skills):
        return 100.0 * sum(1 for r in reqs if token_alias(r, skills)) / len(reqs)

    def scorer_kg(skills):
        return calculate_pipeline_match_score(skills, reqs)[0]

    scorers = {"A_exact": scorer_exact, "B_alias": scorer_alias, "D_kg": scorer_kg,
               "E_kg_shap": scorer_kg, "F_full": scorer_kg}

    tfidf_scores = None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(texts + [jd_text])
        tfidf_scores = [round(float(x) * 100, 4) for x in cosine_similarity(mat[:-1], mat[-1]).flatten()]
    except Exception as e:
        print(f"[WARN] TF-IDF baseline unavailable: {e}")

    configs = {}
    per_case = []
    lat = defaultdict(list)
    scores_by_config = {}
    for name, fn in scorers.items():
        scores = []
        for i, skills in enumerate(skills_per):
            t0 = time.perf_counter()
            if name == "F_full" and AIHiringGuardrails.scan_document(texts[i])["blocked"]:
                s = -1.0  # blocked before scoring: counts as a rejection
            else:
                s = fn(skills)
            lat[name].append((time.perf_counter() - t0) * 1000)
            scores.append(s)
            per_case.append({"config": name, "resume_id": i, "category": resumes[i].get("category"),
                             "relevant_proxy": labels[i], "score_pct": s})
        scores_by_config[name] = scores
        m = confusion(scores, labels, DECISION_THRESHOLD)
        m["auc"] = auc([s for s, y in zip(scores, labels) if y], [s for s, y in zip(scores, labels) if not y])
        m["mean_latency_ms"] = round(sum(lat[name]) / len(lat[name]), 4)
        configs[name] = m
    if tfidf_scores is not None:
        m = confusion(tfidf_scores, labels, TFIDF_THRESHOLD)
        m["auc"] = auc([s for s, y in zip(tfidf_scores, labels) if y], [s for s, y in zip(tfidf_scores, labels) if not y])
        m["mean_latency_ms"] = None
        m["threshold_used"] = TFIDF_THRESHOLD
        configs["C_tfidf"] = m
        scores_by_config["C_tfidf"] = tfidf_scores

    # ---------- skill-case accuracy per scorer ----------
    skill_cases = load_skill_cases()
    skill_results = {}
    for name in ["A_exact", "B_alias", "D_kg"]:
        per_set = defaultdict(lambda: [0, 0])
        for c in skill_cases:
            cand_csv = ", ".join(c["candidate_skills"])
            if name == "A_exact":
                pred = token_exact(c["required_skill"], c["candidate_skills"])
            elif name == "B_alias":
                pred = token_alias(c["required_skill"], c["candidate_skills"])
            else:
                pred = evaluate_skill_match(c["required_skill"], c["candidate_skills"])["is_approved"]
            ok = (pred == (c["expected_label"] == "APPROVED"))
            per_set[c["set"]][0] += int(ok)
            per_set[c["set"]][1] += 1
            per_set["ALL"][0] += int(ok)
            per_set["ALL"][1] += 1
        skill_results[name] = {k: {"correct": v[0], "total": v[1],
                                   "accuracy_pct": round(100 * v[0] / v[1], 2) if v[1] else None}
                               for k, v in per_set.items()}

    # ---------- explanation quality (E and F share the explainer) ----------
    baselines, base_meta = compute_cohort_empirical_baselines(reqs)
    max_add_err = 0.0
    aopc_comp, aopc_suff, eligible, faith_pass = [], [], 0, 0
    for skills in skills_per:
        score, details = calculate_pipeline_match_score(skills, reqs)
        rshap, total, expected = compute_requirement_shap_values(details, reqs, baselines)
        max_add_err = max(max_add_err, abs(total - (score - expected)))
        sshap, _ = compute_skill_attribution_shapley(skills, reqs)
        fr = evaluate_feature_perturbation_faithfulness(skills, sshap, reqs)
        if len(skills) >= 2:
            eligible += 1
            aopc_comp.append(fr["aopc_comprehensiveness"])
            aopc_suff.append(fr["aopc_sufficiency"])
            faith_pass += int(fr["faithfulness_passed"])
    explanation = {
        "coverage_pct": 100.0,
        "max_additivity_error": max_add_err,
        "eligible_candidates": eligible,
        "mean_aopc_comprehensiveness": round(sum(aopc_comp) / len(aopc_comp), 4) if aopc_comp else None,
        "mean_aopc_sufficiency": round(sum(aopc_suff) / len(aopc_suff), 4) if aopc_suff else None,
        "faithfulness_pass_rate_pct": round(100 * faith_pass / eligible, 2) if eligible else None,
        "baseline_metadata": base_meta,
    }

    # ---------- security through the PRODUCTION guardrail ----------
    sec_sets = load_security_sets()
    security = {}
    for set_name, cases in sec_sets.items():
        tp = fp = tn = fn = 0
        for c in cases:
            blocked = AIHiringGuardrails.scan_document(c["text"])["blocked"]
            if c["is_attack"] and blocked: tp += 1
            elif c["is_attack"]: fn += 1
            elif blocked: fp += 1
            else: tn += 1
        security[set_name] = {
            "n_attacks": tp + fn, "n_benign": fp + tn, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "mitigation_rate_pct": round(100 * tp / (tp + fn), 2) if tp + fn else None,
            "false_positive_rate_pct": round(100 * fp / (fp + tn), 2) if fp + tn else None,
        }
    # FPR of the production guardrail on the real Kaggle resumes themselves
    kaggle_blocked = sum(1 for t in texts if AIHiringGuardrails.scan_document(t)["blocked"])
    security["kaggle_real_resumes"] = {"n": len(texts), "blocked": kaggle_blocked,
                                       "false_positive_rate_pct": round(100 * kaggle_blocked / len(texts), 2)}

    # ---------- ablation summary ----------
    ablation = {
        "kg_vs_exact_resume_f1_delta": round(configs["D_kg"]["f1_score"] - configs["A_exact"]["f1_score"], 4),
        "kg_vs_tfidf_resume_auc_delta": (round(configs["D_kg"]["auc"] - configs["C_tfidf"]["auc"], 4) if "C_tfidf" in configs else None),
        "shap_cost_resume_f1_delta": round(configs["E_kg_shap"]["f1_score"] - configs["D_kg"]["f1_score"], 4),
        "shap_adds": "explanation coverage 100%, additivity and AOPC faithfulness as reported; scores identical to D_kg by construction",
        "guardrail_cost_on_clean_resume_f1_delta": round(configs["F_full"]["f1_score"] - configs["E_kg_shap"]["f1_score"], 4),
        "guardrail_benefit": "attack documents reaching the scoring stage fall from 100% (unguarded configs) to (100 - mitigation)% per security set; clean-resume scores are unchanged unless a real resume is falsely blocked (see kaggle_real_resumes)",
    }

    out = {
        "metadata": {
            "runner": "run_integrated_benchmarks.py", "seed": 42,
            "decision_threshold_pct": DECISION_THRESHOLD,
            "requirements": reqs, "n_resumes": len(resumes),
            "proxy_label": "category != 'Accountant' (no human hiring ground truth in corpus)",
            "guardrail_under_test": "security_guardrails.AIHiringGuardrails.scan_document (production, used by api.py)",
            "elapsed_sec": round(time.time() - t_start, 2),
        },
        "resume_screening_by_config": configs,
        "skill_case_accuracy": skill_results,
        "explanation_quality_E_and_F": explanation,
        "security_production_guardrail": security,
        "ablation_summary": ablation,
    }
    with open(os.path.join(DATA, "integrated_benchmark_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(DATA, "integrated_per_case_results.json"), "w") as f:
        json.dump(per_case, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
