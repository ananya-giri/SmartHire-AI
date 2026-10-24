"""
SmartHire AI: Deterministic Knowledge Graph SHAP Scoring Explainer
===================================================================
Connects game-theoretic SHAP explainability directly to the deterministic 
Neuro-Symbolic Knowledge Graph candidate matching engine (Graph_ATS).

Important Architectural Boundary Notice:
1. Production LLM Screener: The primary candidate fit score in the multi-agent
   recruitment pipeline is generated as unstructured natural language markdown by
   an LLM prompt (`screening_agent`). That generative narrative is probabilistic
   and cannot be faithfully explained by post-hoc Shapley values without surrogate
   distortions.
2. Deterministic Knowledge Graph Matching: This module provides game-theoretic
   feature attributions for the separate, deterministic candidate-matching engine
   (`calculate_pipeline_match_score`), which evaluates candidate skills against
   job requirements using exact word-boundary matching, industry aliases, and
   Dijkstra shortest-path distances (d <= 2) on the technological Knowledge Graph.

Methodological Distinctions:
- Algebraic Additivity: sum(phi_i) == Score - E[Score]. An exact algebraic identity
  guaranteed by the closed-form Shapley formulation for linearly separable games.
  Additivity alone is NOT proof of causal or empirical faithfulness.
- Feature-Perturbation Faithfulness: Tested empirically by comparing score degradation
  when removing top-contributing features (Comprehensiveness) vs null-player features
  (Sufficiency) via Area Over the Perturbation Curve (AOPC).
- Predictive Validity: Requires human ground-truth hiring outcomes (which are absent
  in open-source resume corpora).
"""

import os
import sys
import re
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Union

# Ensure path to tools
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tools.knowledge_graph_tool import (
    evaluate_skill_match,
    _GLOBAL_TECH_GRAPH,
    SKILL_ALIASES,
    _normalize_skill_name
)

# Standard benchmark requirements (Senior Full-Stack Developer)
DEFAULT_JD_REQUIREMENTS = [
    "React",
    "TypeScript",
    "Node.js",
    "Python",
    "FastAPI",
    "PostgreSQL",
    "MongoDB",
    "Docker",
    "AWS",
    "CI/CD"
]


def exact_keyword_match(required_skill: str, candidate_text: str) -> bool:
    """
    Case-insensitive exact word/phrase boundary match.
    Uses regex word boundaries to prevent accidental substring matches
    (e.g., 'Java' matching inside 'JavaScript', or 'Go' matching inside 'Django').
    """
    if not required_skill or not candidate_text:
        return False
    clean_req = re.escape(required_skill.strip())
    pattern = rf"(?i)(?:\b|_){clean_req}(?:\b|_)"
    return bool(re.search(pattern, candidate_text))


def alias_normalized_match(required_skill: str, candidate_text: str) -> bool:
    """
    Matches known technology aliases against candidate text using word boundaries.
    Prevents false containment matches while recognizing canonical shorthand
    (e.g., 'k8s' -> Kubernetes, 'py' -> Python, 'postgres' -> PostgreSQL).
    """
    if not required_skill or not candidate_text:
        return False
    graph_nodes = set(_GLOBAL_TECH_GRAPH.nodes)
    req_mapped = _normalize_skill_name(required_skill, graph_nodes)
    
    for alias_raw, canonical in SKILL_ALIASES.items():
        if canonical.lower() == req_mapped.lower():
            if exact_keyword_match(alias_raw, candidate_text):
                return True
    return False


def extract_candidate_skill_tokens(text: str) -> List[str]:
    """
    Extracts distinct technological skill tokens from text by matching
    against the Knowledge Graph ontology and standard tech nomenclature.
    Uses strict word-boundary matching.
    """
    if not text:
        return []

    graph_nodes = set(_GLOBAL_TECH_GRAPH.nodes)
    found_skills = set()

    # 1. Check graph canonical names
    for node in graph_nodes:
        if exact_keyword_match(node, text):
            found_skills.add(node)

    # 2. Check known aliases
    for alias_raw, canonical in SKILL_ALIASES.items():
        if exact_keyword_match(alias_raw, text):
            found_skills.add(canonical)

    # 3. Add common standardized tech tokens
    generic_tech_tokens = [
        "Git", "SVN", "Jira", "Webpack", "Babel", "Jest", "Enzyme", "Cypress",
        "HTML", "CSS", "SASS", "Redux", "Zustand", "Linux", "Unix", "Bash",
        "Agile", "Scrum", "TDD", "CI/CD", "REST", "JSON", "XML", "GraphQL",
        "SQL", "Redis", "Kafka", "RabbitMQ", "Kubernetes", "Docker", "AWS", "GCP", "Azure"
    ]
    for gt in generic_tech_tokens:
        if exact_keyword_match(gt, text):
            found_skills.add(gt)

    return sorted(list(found_skills))


def extract_job_requirements(jd_text: str) -> List[str]:
    """
    Extracts structured technical job requirements from an arbitrary Job Description text.
    Uses Knowledge Graph ontology nodes, industry aliases, and technical nomenclatures.
    Returns empty list if no technical competencies are detected.
    """
    return extract_candidate_skill_tokens(jd_text)


def normalize_required_skills(required_skills) -> List[str]:
    """Gap-1 fix: accept a JD string, a comma-separated string, or a list.

    Before this fix, main.py passed the raw job-description text as
    required_skills; the scorer then iterated over its characters and the
    resulting 'explanation' was meaningless. A string is now always treated
    as job-description text and parsed with extract_job_requirements().
    """
    if required_skills is None:
        return list(DEFAULT_JD_REQUIREMENTS)
    if isinstance(required_skills, str):
        tokens = [s.strip() for s in required_skills.replace("\n", ",").split(",") if s.strip()]
        if len(tokens) <= 12 and all(len(t.split()) <= 3 for t in tokens):
            return tokens
        extracted = extract_job_requirements(required_skills)
        return extracted if extracted else list(DEFAULT_JD_REQUIREMENTS)
    return [str(s).strip() for s in required_skills if str(s).strip()]


def calculate_pipeline_match_score(
    candidate_text_or_skills: Union[str, List[str]],
    required_skills: List[str] = None
) -> Tuple[float, Dict[str, Dict[str, Any]]]:
    """
    Executes the deterministic Knowledge Graph candidate matching pipeline (Graph_ATS).
    
    Rules for Approval:
      1. Word-boundary exact match (exact_keyword_match) -> EXACT (d=0)
      2. Normalized industry alias (alias_normalized_match) -> ALIAS (d=0)
      3. Shortest path on graph <= 2 (evaluate_skill_match) -> DIRECT_SUBSUMPTION (d=1) or TRANSFERABLE (d=2)
      
    NOTE: Accidental raw substring containment ('req.lower() in text.lower()') is strictly prohibited.
    
    Returns:
      - score_pct: float in [0.0, 100.0]
      - match_details: Dict mapping each requirement to its approval status,
        match type (EXACT, ALIAS, DIRECT_SUBSUMPTION, TRANSFERABLE, GAP),
        graph shortest-path distance, and closest skill.
    """
    required_skills = normalize_required_skills(required_skills)

    if isinstance(candidate_text_or_skills, (list, set, tuple)):
        cand_tokens = [str(s).strip() for s in candidate_text_or_skills if str(s).strip()]
        cand_str = ", ".join(cand_tokens)
    else:
        cand_str = str(candidate_text_or_skills)
        cand_tokens = [s.strip() for s in cand_str.replace(";", ",").replace("\n", ",").split(",") if s.strip()]

    matched_count = 0
    match_details = {}

    for req in required_skills:
        m_exact = exact_keyword_match(req, cand_str)
        m_alias = alias_normalized_match(req, cand_str)
        kg_eval = evaluate_skill_match(req, cand_tokens if cand_tokens else cand_str)
        
        is_approved = bool(m_exact or m_alias or kg_eval.get("is_approved", False))
        
        if is_approved:
            matched_count += 1
            if m_exact:
                match_type = "EXACT"
                dist = 0.0
                closest = req
            elif m_alias:
                match_type = "ALIAS"
                dist = 0.0
                closest = kg_eval.get("closest_skill") or req
            else:
                match_type = kg_eval.get("match_type", "GRAPH_MATCH")
                dist = kg_eval.get("weighted_distance", 1.0)
                closest = kg_eval.get("closest_skill")
        else:
            match_type = "GAP"
            dist = float("inf")
            closest = None
            
        match_details[req] = {
            "is_approved": is_approved,
            "match_type": match_type,
            "distance": dist,
            "closest_skill": closest
        }

    total_reqs = len(required_skills)
    score_pct = (matched_count / total_reqs) * 100.0 if total_reqs > 0 else 0.0
    return round(score_pct, 4), match_details


# Empirical observed requirement satisfaction baselines over benchmark Kaggle resumes (N=20)
DEFAULT_OBSERVED_BASELINES = {
    "React": 0.45,
    "TypeScript": 0.45,
    "Node.js": 0.70,
    "Python": 0.70,
    "FastAPI": 0.60,
    "PostgreSQL": 0.55,
    "MongoDB": 0.55,
    "Docker": 0.60,
    "AWS": 0.50,
    "CI/CD": 0.60
}
EMPIRICAL_BENCHMARK_BASELINES = DEFAULT_OBSERVED_BASELINES


def compute_cohort_empirical_baselines(
    required_skills: List[str],
    cohort_resumes: List[Dict[str, Any]] = None
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Calculates actual empirical requirement baseline satisfaction rates p_j = E[Z_j]
    from an observed resume cohort (defaults to data/extracted_kaggle_resumes.json).
    
    Returns:
      - baselines: Dict[str, float] mapping each required skill to its observed rate in [0.0, 1.0].
      - metadata: Dict[str, Any] detailing cohort size, source file, and calculation provenance.
    """
    if cohort_resumes is None:
        kaggle_path = os.path.join(BASE_DIR, "data", "extracted_kaggle_resumes.json")
        if os.path.exists(kaggle_path):
            try:
                with open(kaggle_path, "r", encoding="utf-8") as f:
                    cohort_resumes = json.load(f)
            except Exception:
                cohort_resumes = []
        else:
            cohort_resumes = []

    N = len(cohort_resumes)
    baselines = {}
    
    if N > 0:
        counts = {r: 0 for r in required_skills}
        for cand in cohort_resumes:
            c_text = cand.get("text", "")
            c_skills = extract_candidate_skill_tokens(c_text)
            _, det = calculate_pipeline_match_score(c_skills, required_skills)
            for r in required_skills:
                if det[r]["is_approved"]:
                    counts[r] += 1
        baselines = {r: round(counts[r] / N, 4) for r in required_skills}
        metadata = {
            "source": "data/extracted_kaggle_resumes.json",
            "cohort_size": N,
            "provenance": "Observed empirical pass rates over authentic Kaggle resume cohort",
            "observed_rates": baselines
        }
    else:
        # Neutral reference baseline if no cohort is available
        baselines = {r: 0.50 for r in required_skills}
        metadata = {
            "source": "uninformative_prior",
            "cohort_size": 0,
            "provenance": "Neutral reference expectation prior (0.50) due to unsupplied cohort",
            "observed_rates": baselines
        }

    return baselines, metadata


def compute_requirement_shap_values(
    match_details: Dict[str, Dict[str, Any]],
    required_skills: List[str] = None,
    baseline_rates: Dict[str, float] = None
) -> Tuple[Dict[str, Dict[str, Any]], float, float]:
    """
    Computes closed-form game-theoretic Shapley attributions over required job competencies.
    
    Game Definition:
      Binary approval state z_j in {0, 1}.
      Baseline expectation p_j in [0, 1].
      Score(z) = (100 / M) * sum_{j=1}^M z_j
      E[Score] = (100 / M) * sum_{j=1}^M p_j
      phi_j = (100 / M) * (z_j - p_j)
      
    Algebraic Efficiency Identity:
      sum_{j=1}^M phi_j == Score - E[Score] (error = 0.00e+00)
    """
    if required_skills is None:
        required_skills = list(match_details.keys())
    if baseline_rates is None:
        baseline_rates, _ = compute_cohort_empirical_baselines(required_skills)

    M = len(required_skills)
    if M == 0:
        return {}, 0.0, 0.0

    shap_records = {}
    total_shap = 0.0
    expected_score = sum(baseline_rates.get(r, 0.50) for r in required_skills) * (100.0 / M)

    for req in required_skills:
        info = match_details.get(req, {"is_approved": False, "match_type": "GAP", "distance": float('inf')})
        z_j = 1.0 if info["is_approved"] else 0.0
        p_j = baseline_rates.get(req, 0.50)
        
        phi_j = (100.0 / M) * (z_j - p_j)
        total_shap += phi_j
        
        shap_records[req] = {
            "is_approved": bool(z_j),
            "match_type": info["match_type"],
            "distance": info["distance"],
            "closest_skill": info.get("closest_skill"),
            "shap_value_pct": round(phi_j, 4),
            "baseline_expectation_pct": round(p_j * 100.0, 1),
            "contribution_type": "POSITIVE_BOOST" if phi_j > 0 else "GAP_PENALTY"
        }

    return shap_records, round(total_shap, 6), round(expected_score, 6)


def compute_skill_attribution_shapley(
    candidate_skills: List[str],
    required_skills: List[str] = None
) -> Tuple[Dict[str, float], float]:
    """
    Computes exact closed-form Shapley values over candidate's reported skills in the matching game.
    
    Characteristic Function:
      v(U) = calculate_pipeline_match_score(U, required_skills)[0] for coalition U subset candidate_skills.
      
    Closed-Form Solution:
      In this separable 1-out-of-k threshold game:
      For requirement r_j, let C_j = {s in candidate_skills : s satisfies r_j}.
      Each skill s in C_j receives equal marginal share (100 / M) / |C_j|.
      Skills not in C_j receive 0.0 (Null Player Axiom).
      phi(s) = sum_{j : s in C_j} (100 / M) / |C_j|
      
    Guaranteed Axioms:
      - Efficiency: sum(phi(s_k)) == Score(candidate_skills) - Score(empty) == Score.
      - Null Player (Dummy): Unmatched tokens receive strictly 0.0000.
      - Symmetry: Interchangeable skills satisfying identical requirements receive equal credit.
    """
    required_skills = normalize_required_skills(required_skills)

    skills = list(candidate_skills)
    n = len(skills)
    if n == 0:
        return {}, 0.0

    M = len(required_skills)
    if M == 0:
        return {s: 0.0 for s in skills}, 0.0

    skill_req_incidence = {}
    for s in skills:
        _, det = calculate_pipeline_match_score([s], required_skills)
        skill_req_incidence[s] = {r: det[r]["is_approved"] for r in required_skills}

    shap_values = {s: 0.0 for s in skills}
    for req in required_skills:
        satisfying_skills = [s for s in skills if skill_req_incidence[s][req]]
        k_satisfying = len(satisfying_skills)
        if k_satisfying > 0:
            marginal_share = (100.0 / M) / k_satisfying
            for s in satisfying_skills:
                shap_values[s] += marginal_share

    total_shap = sum(shap_values.values())
    return {s: round(v, 4) for s, v in shap_values.items()}, round(total_shap, 4)


def evaluate_feature_perturbation_faithfulness(
    candidate_skills: List[str],
    skill_shap_values: Dict[str, float],
    required_skills: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluates Feature-Perturbation Faithfulness via Area Over the Perturbation Curve (AOPC).
    
    Compares:
      1. Comprehensiveness (Top-Feature Removal): Ablating top-contributing skills
         must cause a rapid, monotonic score decay.
      2. Sufficiency (Bottom-Feature Removal): Ablating bottom/null skills
         must preserve the score with zero or minimal decay.
         
    Faithful Explanation Standard:
      AOPC_comprehensiveness >= AOPC_sufficiency
    """
    required_skills = normalize_required_skills(required_skills)

    baseline_score = calculate_pipeline_match_score(candidate_skills, required_skills)[0]
    
    sorted_skills_desc = [s for s, v in sorted(skill_shap_values.items(), key=lambda x: x[1], reverse=True)]
    sorted_skills_asc = [s for s, v in sorted(skill_shap_values.items(), key=lambda x: x[1])]

    k_max = min(len(candidate_skills), 5)
    if k_max == 0:
        return {
            "baseline_score": baseline_score,
            "k_evaluated": 0,
            "aopc_comprehensiveness": 0.0,
            "aopc_sufficiency": 0.0,
            "faithfulness_ratio": 1.0,
            "faithfulness_passed": True,
            "top_feature_decay": [],
            "bottom_feature_decay": []
        }

    comp_scores = []
    suff_scores = []

    # 1. Top-k removal
    for k in range(1, k_max + 1):
        ablated = [s for s in candidate_skills if s not in sorted_skills_desc[:k]]
        sc = calculate_pipeline_match_score(ablated, required_skills)[0]
        comp_scores.append(baseline_score - sc)

    # 2. Bottom-k removal
    for k in range(1, k_max + 1):
        ablated = [s for s in candidate_skills if s not in sorted_skills_asc[:k]]
        sc = calculate_pipeline_match_score(ablated, required_skills)[0]
        suff_scores.append(baseline_score - sc)

    aopc_comp = float(np.mean(comp_scores)) if comp_scores else 0.0
    aopc_suff = float(np.mean(suff_scores)) if suff_scores else 0.0
    ratio = aopc_comp / (aopc_suff + 1e-4)

    is_faithful = bool(aopc_comp >= aopc_suff)

    return {
        "baseline_score": baseline_score,
        "k_evaluated": k_max,
        "aopc_comprehensiveness": round(aopc_comp, 4),
        "aopc_sufficiency": round(aopc_suff, 4),
        "faithfulness_ratio": round(ratio, 2),
        "faithfulness_passed": is_faithful,
        "top_feature_decay": [round(x, 2) for x in comp_scores],
        "bottom_feature_decay": [round(x, 2) for x in suff_scores]
    }


def generate_pipeline_shap_report(
    candidate_text_or_skills: Union[str, List[str]],
    required_skills: List[str] = None,
    candidate_name: str = "Candidate",
    baseline_rates: Dict[str, float] = None
) -> str:
    """
    Generates a structured SHAP audit report directly coupled to the 
    Knowledge Graph candidate matching pipeline.
    """
    required_skills = normalize_required_skills(required_skills)

    # 1. Pipeline Match Score
    score_pct, match_details = calculate_pipeline_match_score(candidate_text_or_skills, required_skills)

    # 2. Baseline rates
    if baseline_rates is None:
        baseline_rates, base_meta = compute_cohort_empirical_baselines(required_skills)
    else:
        base_meta = {"source": "caller_specified", "provenance": "Supplied baseline dictionary"}

    # 3. Requirement-Level SHAP
    req_shap, total_req_shap, base_expected = compute_requirement_shap_values(
        match_details, required_skills, baseline_rates
    )

    # 4. Candidate-Skill SHAP
    if isinstance(candidate_text_or_skills, (list, set, tuple)):
        cand_skills = list(candidate_text_or_skills)
    else:
        cand_skills = extract_candidate_skill_tokens(str(candidate_text_or_skills))

    skill_shap, total_skill_shap = compute_skill_attribution_shapley(cand_skills, required_skills)

    # 5. Perturbation Faithfulness Check
    faith_results = evaluate_feature_perturbation_faithfulness(cand_skills, skill_shap, required_skills)

    lines = []
    lines.append("=" * 75)
    lines.append(f"SMARTHIRE AI: KNOWLEDGE GRAPH SHAP EXPLAINABILITY DOSSIER ({candidate_name})")
    lines.append("=" * 75)
    lines.append(f"Deterministic Match Score: {score_pct:.1f}%")
    lines.append(f"Cohort Expectation E[Score]: {base_expected:.1f}% (Baseline Source: {base_meta.get('source')})")
    lines.append(f"Attribution Shift (Score - E[Score]): {score_pct - base_expected:+.1f}%\n")

    lines.append("--- PART 1: REQUIREMENT-LEVEL SHAP VALUES (phi_j) ---")
    lines.append("Measures marginal push above or below cohort expectation for each requirement:")
    
    sorted_reqs = sorted(req_shap.items(), key=lambda x: x[1]["shap_value_pct"], reverse=True)
    for req, rdata in sorted_reqs:
        v = rdata["shap_value_pct"]
        sign = "+" if v > 0 else ""
        mtype = rdata["match_type"]
        dist = rdata["distance"]
        closest = f" (via '{rdata['closest_skill']}')" if rdata.get("closest_skill") and mtype != "EXACT" else ""
        lines.append(f" - {req:<12}: {sign}{v:+.2f}% | Status: {mtype}{closest} [d={dist}]")

    additivity_error = abs(total_req_shap - (score_pct - base_expected))
    lines.append(f"\nAlgebraic Additivity Check: sum(phi_j) = {total_req_shap:.2f}%, Expected Shift = {score_pct - base_expected:.2f}%, Error = {additivity_error:.8f}")

    lines.append("\n--- PART 2: CANDIDATE SKILL SHAPLEY ATTRIBUTIONS ---")
    lines.append(f"Coalition game over {len(cand_skills)} candidate skills:")
    sorted_skills = sorted(skill_shap.items(), key=lambda x: x[1], reverse=True)
    for s, v in sorted_skills[:12]:
        sign = "+" if v > 0 else ""
        lines.append(f" - {s:<18}: {sign}{v:.2f}% marginal contribution")
    
    skill_additivity_error = abs(total_skill_shap - score_pct)
    lines.append(f"\nSkill Additivity Sum: {total_skill_shap:.2f}% | Actual Score: {score_pct:.2f}% | Error: {skill_additivity_error:.8f}")

    lines.append("\n--- PART 3: FEATURE-PERTURBATION FAITHFULNESS AUDIT (AOPC) ---")
    lines.append(f"Comprehensiveness Decay (Top-k removal): {faith_results['aopc_comprehensiveness']:.2f}% drop")
    lines.append(f"Sufficiency Decay (Bottom-k removal):    {faith_results['aopc_sufficiency']:.2f}% drop")
    lines.append(f"Faithfulness Status:                     {'PASSED (Comprehensiveness >= Sufficiency)' if faith_results['faithfulness_passed'] else 'FAILED'}")

    lines.append("\nArchitectural Scope & Limitations:")
    lines.append("1. This explainer strictly interprets the deterministic Neuro-Symbolic Knowledge Graph")
    lines.append("   matching pipeline. It does NOT explain unstructured natural language reasoning")
    lines.append("   generated by multi-agent LLM prompts.")
    lines.append("2. Exact additivity is an algebraic identity of the closed-form formulation. It does not")
    lines.append("   guarantee predictive validity against human hiring outcomes without empirical outcome data.")
    lines.append("=" * 75)

    return "\n".join(lines)
