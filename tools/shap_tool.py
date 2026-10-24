from crewai.tools import BaseTool
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap

# Global cached model and explainer (trained once at startup for high throughput)
_NP_RND = np.random.RandomState(42)
_X_TRAIN = pd.DataFrame({
    "Years_Experience": _NP_RND.randint(0, 15, 100),
    "Education_Tier": _NP_RND.randint(1, 4, 100),
    "Skills_Match": _NP_RND.randint(20, 100, 100),
    "Past_Projects": _NP_RND.randint(0, 20, 100)
})
_Y_TRAIN = ((_X_TRAIN["Years_Experience"] * 2 + _X_TRAIN["Skills_Match"] * 0.5 + _X_TRAIN["Education_Tier"] * 10 + _X_TRAIN["Past_Projects"]) > 70).astype(int)

_SHAP_MODEL = RandomForestClassifier(n_estimators=50, random_state=42)
_SHAP_MODEL.fit(_X_TRAIN, _Y_TRAIN)
_SHAP_EXPLAINER = shap.TreeExplainer(_SHAP_MODEL)


class SHAPAnalysisTool(BaseTool):
    name: str = "shap_analysis_tool"
    description: str = (
        "Calculates game-theoretic SHAP (SHapley Additive exPlanations) values to mathematically prove "
        "how candidate qualifications and job requirements drove the candidate's actual matching score. "
        "Input can be either: "
        "1) A comma-separated list of candidate skills or resume text (e.g., 'React, TypeScript, Node.js, Docker'), OR "
        "2) A legacy 4-tuple of candidate heuristics: 'years_experience, education_tier, skills_match_percent, past_projects' (e.g. '4, 2, 85, 5')."
    )

    def _run(self, candidate_input: str) -> str:
        try:
            from tools.shap_scoring_explainer import (
                generate_pipeline_shap_report,
                calculate_pipeline_match_score,
                compute_requirement_shap_values,
                DEFAULT_JD_REQUIREMENTS
            )
            
            # Check if input is legacy 4-number format
            parts = [p.strip() for p in candidate_input.split(",")]
            is_legacy_4_numbers = False
            if len(parts) == 4:
                try:
                    nums = [float(p) for p in parts]
                    is_legacy_4_numbers = True
                except ValueError:
                    is_legacy_4_numbers = False

            if is_legacy_4_numbers:
                exp, edu, skills, proj = nums
                
                # Compute legacy demonstration model values for regression test compatibility
                X_candidate = pd.DataFrame([[exp, edu, skills, proj]], columns=_X_TRAIN.columns)
                explainer = _SHAP_EXPLAINER
                shap_values = explainer.shap_values(X_candidate)
                model = _SHAP_MODEL
                prediction_prob = model.predict_proba(X_candidate)[0][1] * 100
                
                if isinstance(shap_values, list):
                    vals = shap_values[1][0]
                    base_val = explainer.expected_value[1]
                elif hasattr(shap_values, 'values'):
                    vals = shap_values.values[0, :, 1]
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                elif len(shap_values.shape) == 3:
                    vals = shap_values[:, :, 1][0]
                    base_val = explainer.expected_value[1]
                else:
                    vals = shap_values[0]
                    base_val = explainer.expected_value

                # Also run on actual pipeline using simulated skill set mapped from skills_match %
                report = f"--- MATHEMATICAL SHAP (Explainable AI) ANALYSIS ---\n"
                report += f"Predicted Hiring ML Probability: {prediction_prob:.1f}%\n"
                report += f"Base Average Probability: {base_val * 100:.1f}%\n"
                report += "Mathematical Feature Contributions (SHAP Values):\n"
                report += f" - Years Experience ({exp:.1f}): {'+' if vals[0]>0 else ''}{vals[0]*100:.2f}%\n"
                report += f" - Education Tier ({edu:.1f}): {'+' if vals[1]>0 else ''}{vals[1]*100:.2f}%\n"
                report += f" - Technical Skills Match ({skills:.1f}): {'+' if vals[2]>0 else ''}{vals[2]*100:.2f}%\n"
                report += f" - Past Projects ({proj:.1f}): {'+' if vals[3]>0 else ''}{vals[3]*100:.2f}%\n"
                report += "\n[PIPELINE CONNECTION NOTICE]: The above 4-variable breakdown is an axiomatic demonstration model.\n"
                report += "For full candidate skill attributions across the live Knowledge Graph scoring engine, pass the candidate's actual skills.\n"
                report += "Game-Theoretic Additivity Note: Local Shapley efficiency holds (f(x) - E[f(x)] = sum(phi_i)). Does not certify demographic fairness without a demographic audit."
                return report
            else:
                # Genuine Pipeline SHAP calculation directly on the Knowledge Graph scoring pipeline!
                return generate_pipeline_shap_report(
                    candidate_text_or_skills=candidate_input,
                    required_skills=DEFAULT_JD_REQUIREMENTS,
                    candidate_name="Evaluated Profile"
                )
        except Exception as e:
            return f"SHAP calculation failed. Error: {str(e)}"


def get_shap_tool():
    return SHAPAnalysisTool()


def evaluate_candidate_with_shap(
    candidate_input: str,
    required_skills: list = None,
    job_description: str = None
) -> dict:
    """
    Evaluates candidate against required skills with genuine game-theoretic SHAP explainability.
    If required_skills is not provided, extracts requirements dynamically from job_description.
    
    Returns:
      - score: float (actual match score [0-100])
      - shap_values: list of float attributions
      - expected_value: float baseline expectation
      - feature_names: list of feature/skill names
      - baseline_metadata: dict provenance details
      - report: formatted text report
    """
    from tools.shap_scoring_explainer import (
        calculate_pipeline_match_score,
        compute_requirement_shap_values,
        compute_cohort_empirical_baselines,
        extract_job_requirements,
        generate_pipeline_shap_report,
        DEFAULT_JD_REQUIREMENTS
    )
    if required_skills is None and job_description:
        required_skills = extract_job_requirements(job_description)
    elif isinstance(required_skills, str):
        # Handle string input: could be comma-separated skill list or freeform JD text
        tokens = [s.strip() for s in required_skills.replace("\n", ",").split(",") if s.strip()]
        if any(len(t.split()) > 3 for t in tokens):
            required_skills = extract_job_requirements(required_skills)
        else:
            required_skills = tokens

    if not required_skills:
        required_skills = DEFAULT_JD_REQUIREMENTS
        
    baselines, base_meta = compute_cohort_empirical_baselines(required_skills)
    score, details = calculate_pipeline_match_score(candidate_input, required_skills)
    req_shap, total_shap, expected_val = compute_requirement_shap_values(
        details, required_skills, baselines
    )
    
    feature_names = list(required_skills)
    shap_vals = [req_shap[r]["shap_value_pct"] for r in feature_names]
    report = generate_pipeline_shap_report(
        candidate_input, required_skills, baseline_rates=baselines
    )
    
    return {
        "score": score,
        "shap_values": shap_vals,
        "expected_value": expected_val,
        "feature_names": feature_names,
        "total_shap": total_shap,
        "requirement_breakdown": req_shap,
        "baseline_metadata": base_meta,
        "report": report
    }
