"""
Active-application verification tests (Gap-2 + Gap-1 wiring, v3).

These tests exist because the original suite benchmarked components that
the application did not run. They verify:
  1. The PRODUCTION guardrail (security_guardrails.scan_document, the exact
     function api.py calls) achieves the reported mitigation on the frozen
     dev security set, keeps 0% FPR on benign controls, and blocks none of
     the 20 real Kaggle resumes.
  2. The live pipeline wiring: resume PROSE + JD PROSE -> extracted
     requirements/skills -> deterministic score + Shapley report, including
     the regression case that broke v2 (JD string passed as requirements).
  3. Static wiring: api.py exposes /score and /guardrail/status and calls
     scan_document; main.py contains no fabricated fallback percentages;
     benchmark_suite delegates its Layer-2 to the production module.

Run: python -m unittest test_active_guardrail_and_pipeline -v
Requires only networkx + numpy for the deterministic stack.
"""
import json
import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
DATA = os.path.join(BASE_DIR, "data")


class TestProductionGuardrail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from security_guardrails import AIHiringGuardrails
        cls.G = AIHiringGuardrails
        dev = json.load(open(os.path.join(DATA, "per_case_results.json")))
        cls.dev_sec = [c for c in dev if c.get("experiment") == "Security_Guardrails"]

    def test_dev_mitigation_and_fpr(self):
        tp = fn = fp = tn = 0
        for c in self.dev_sec:
            blocked = self.G.scan_document(c["text"])["blocked"]
            if c["is_attack"] and blocked: tp += 1
            elif c["is_attack"]: fn += 1
            elif blocked: fp += 1
            else: tn += 1
        mitigation = 100 * tp / (tp + fn)
        fpr = 100 * fp / (fp + tn)
        self.assertGreaterEqual(mitigation, 70.0, f"production mitigation {mitigation}% < 70%")
        self.assertEqual(fpr, 0.0, "production guardrail must not flag benign controls")

    def test_real_resumes_not_blocked(self):
        resumes = json.load(open(os.path.join(DATA, "extracted_kaggle_resumes.json")))
        blocked = [i for i, r in enumerate(resumes) if self.G.scan_document(r["text"])["blocked"]]
        self.assertEqual(blocked, [], f"real resumes falsely blocked: {blocked}")

    def test_status_reports_layers(self):
        status = self.G.guardrail_status()
        for k in ("layer1_regex", "layer2_heuristic", "layer3_structural"):
            self.assertTrue(status[k], f"{k} must be active in production")


class TestPipelineWiring(unittest.TestCase):
    def test_prose_to_score_and_explanation(self):
        from tools.shap_scoring_explainer import (
            extract_job_requirements, extract_candidate_skill_tokens,
            calculate_pipeline_match_score, generate_pipeline_shap_report)
        jd = "We need a Senior Full-Stack Developer with React, TypeScript, Node.js, Python, FastAPI, PostgreSQL, MongoDB, Docker, AWS and CI/CD."
        resume = ("Full-stack engineer building with Next.js, TypeScript and Node.js. "
                  "APIs in Python with FastAPI, data in Postgres and MongoDB. "
                  "Ships with Docker on AWS, CI/CD via GitHub Actions.")
        reqs = extract_job_requirements(jd)
        skills = extract_candidate_skill_tokens(resume)
        self.assertIn("React", reqs)
        score, details = calculate_pipeline_match_score(skills, reqs)
        # Next.js -> React (d=1) and postgres -> PostgreSQL (alias) must approve
        self.assertTrue(details["React"]["is_approved"])
        self.assertTrue(details["PostgreSQL"]["is_approved"])
        self.assertGreaterEqual(score, 90.0)
        report = generate_pipeline_shap_report(skills, reqs)
        self.assertIn("Deterministic Match Score", report)
        self.assertIn("Algebraic Additivity Check", report)

    def test_jd_string_regression(self):
        """v2 bug: JD text passed as required_skills iterated over characters."""
        from tools.shap_scoring_explainer import generate_pipeline_shap_report, normalize_required_skills
        jd = "Looking for a React and Python developer with Docker experience."
        reqs = normalize_required_skills(jd)
        self.assertTrue(all(len(r) > 1 for r in reqs), f"requirements degenerated: {reqs}")
        self.assertIn("React", reqs)
        report = generate_pipeline_shap_report("React, Python, Docker", jd)
        self.assertIn("Deterministic Match Score: 100.0%", report)

    def test_graph_stats(self):
        from tools.knowledge_graph_tool import get_graph_stats
        stats = get_graph_stats()
        self.assertEqual(stats["nodes"], 70)
        self.assertEqual(stats["edges"], 74)
        self.assertFalse(stats["directed"])


class TestStaticWiring(unittest.TestCase):
    def test_api_uses_production_scan(self):
        src = open(os.path.join(BASE_DIR, "api.py")).read()
        self.assertIn("scan_document", src)
        self.assertIn('/guardrail/status', src)
        self.assertIn('/score', src)

    def test_main_has_no_fabricated_fallbacks(self):
        src = open(os.path.join(BASE_DIR, "main.py")).read()
        self.assertNotIn("Core Skills Match (+38%)", src)
        self.assertNotIn("Candidate competencies aligned with required skill architecture", src)
        self.assertIn("extract_job_requirements", src)
        self.assertIn("AUTHORITATIVE", src)

    def test_benchmark_delegates_to_production(self):
        src = open(os.path.join(BASE_DIR, "benchmark_suite.py")).read()
        self.assertIn("from security_guardrails import heuristic_regex_layer_2", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
