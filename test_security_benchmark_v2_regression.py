"""
Regression Test: Security Benchmark v2 Metric Consistency
==========================================================
Asserts that EVERY value in data/security_benchmark_v2.json (aggregate summary)
is bit-for-bit consistent with metrics recomputed bottom-up from
data/security_per_case_v2.json (per-case ground truth records).

Definitions:
  BLOCKED -> positive prediction (attack detected)
  PASSED  -> negative prediction (not flagged)
  is_attack=True  -> ground-truth positive
  is_attack=False -> ground-truth negative

Confusion matrix (canonical, matches benchmark_suite.run_security_benchmark):
  TP = is_attack AND BLOCKED
  FP = NOT is_attack AND BLOCKED
  TN = NOT is_attack AND NOT BLOCKED
  FN = is_attack AND NOT BLOCKED

Derived metrics:
  accuracy_pct            = (TP+TN)/(TP+FP+TN+FN)*100
  precision_pct           = TP/(TP+FP)*100        [0 when undefined]
  recall_mitigation_pct   = TP/(TP+FN)*100        [0 when undefined]
  f1_score                = 2*prec*rec/(prec+rec)  [0 when undefined]
  false_positive_rate_pct = FP/(FP+TN)*100         [0 when undefined]

Per-family:
  total    = cases in that family (across all prediction outcomes)
  detected = cases where prediction == BLOCKED (regardless of is_attack)
  rate_pct = detected / total * 100  (rounded to 1 decimal)

Run with:
    python test_security_benchmark_v2_regression.py
    pytest test_security_benchmark_v2_regression.py -v
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PER_CASE_PATH  = os.path.join(DATA_DIR, "security_per_case_v2.json")
BENCHMARK_PATH = os.path.join(DATA_DIR, "security_benchmark_v2.json")

COMPARATORS = [
    "Regex_Only",
    "Heuristic_Regex_Layer_2",
    "Combined_Dual_Regex",
    "LlamaPromptGuard2",
]
EXPECTED_TOTAL   = 55   # 30 attacks + 25 benign
EXPECTED_ATTACKS = 30
EXPECTED_BENIGN  = 25
EXPECTED_ATTACK_FAMILIES = [
    "Direct_Imperative",
    "Paraphrased_Semantic",
    "Obfuscated_Delimited",
    "Roleplay_SocialEng",
    "Tool_Exfiltration",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _r2(v): return round(v, 2)
def _r1(v): return round(v, 1)
def _r4(v): return round(v, 4)


def load_fixtures():
    assert os.path.exists(PER_CASE_PATH),  "Missing: " + PER_CASE_PATH
    assert os.path.exists(BENCHMARK_PATH), "Missing: " + BENCHMARK_PATH
    with open(PER_CASE_PATH, "r", encoding="utf-8") as fh:
        cases = json.load(fh)
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    return cases, bench


def recompute_metrics(cases):
    """
    Recompute confusion matrices and derived metrics from per-case records.
    Returns (overall_dict, family_rates_dict).
    """
    cm = {c: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for c in COMPARATORS}
    fb = {}  # fam -> comp -> {total, detected}

    for case in cases:
        is_atk = case["is_attack"]
        fam    = case["family"]
        fb.setdefault(fam, {c: {"total": 0, "detected": 0} for c in COMPARATORS})

        for comp in COMPARATORS:
            blocked = (case["predictions"].get(comp, "PASSED") == "BLOCKED")
            if   is_atk and blocked:         cm[comp]["TP"] += 1
            elif not is_atk and blocked:     cm[comp]["FP"] += 1
            elif not is_atk and not blocked: cm[comp]["TN"] += 1
            else:                            cm[comp]["FN"] += 1
            fb[fam][comp]["total"] += 1
            if blocked:
                fb[fam][comp]["detected"] += 1

    overall = {}
    for comp in COMPARATORS:
        tp, fp, tn, fn = cm[comp]["TP"], cm[comp]["FP"], cm[comp]["TN"], cm[comp]["FN"]
        total = tp + fp + tn + fn
        acc  = (tp + tn) / total      if total > 0        else 0.0
        prec = tp / (tp + fp)         if (tp + fp) > 0    else 0.0
        rec  = tp / (tp + fn)         if (tp + fn) > 0    else 0.0
        f1   = 2*prec*rec/(prec+rec)  if (prec + rec) > 0 else 0.0
        fpr  = fp / (fp + tn)         if (fp + tn) > 0    else 0.0
        overall[comp] = {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy_pct":            _r2(acc  * 100),
            "precision_pct":           _r2(prec * 100),
            "recall_mitigation_pct":   _r2(rec  * 100),
            "f1_score":                _r4(f1),
            "false_positive_rate_pct": _r2(fpr  * 100),
        }

    fam_rates = {}
    for fam, cd in fb.items():
        fam_rates[fam] = {}
        for comp, d in cd.items():
            t, det = d["total"], d["detected"]
            fam_rates[fam][comp] = {
                "total":    t,
                "detected": det,
                "rate_pct": _r1((det / t * 100) if t > 0 else 0.0),
            }

    return overall, fam_rates


# ── test functions ─────────────────────────────────────────────────────────────

def test_fixture_files_exist():
    """Both JSON artefacts must be present before any comparison."""
    assert os.path.exists(PER_CASE_PATH),  "Missing: " + PER_CASE_PATH
    assert os.path.exists(BENCHMARK_PATH), "Missing: " + BENCHMARK_PATH


def test_per_case_record_counts():
    """Validate dataset composition: 55 total (30 attack + 25 benign)."""
    cases, _ = load_fixtures()
    assert len(cases) == EXPECTED_TOTAL, (
        "Expected %d total cases, got %d" % (EXPECTED_TOTAL, len(cases))
    )
    attacks = [c for c in cases if c["is_attack"]]
    benigns = [c for c in cases if not c["is_attack"]]
    assert len(attacks) == EXPECTED_ATTACKS, (
        "Expected %d attacks, got %d" % (EXPECTED_ATTACKS, len(attacks))
    )
    assert len(benigns) == EXPECTED_BENIGN, (
        "Expected %d benign, got %d" % (EXPECTED_BENIGN, len(benigns))
    )
    for fam in EXPECTED_ATTACK_FAMILIES:
        n = sum(1 for c in attacks if c["family"] == fam)
        assert n == 6, "Family %s: expected 6 attack cases, got %d" % (fam, n)
    n_ben = sum(1 for c in benigns if c["family"] == "Benign_Control")
    assert n_ben == EXPECTED_BENIGN, (
        "Benign_Control: expected %d, got %d" % (EXPECTED_BENIGN, n_ben)
    )


def test_all_per_case_records_have_required_fields():
    """Every record must have mandatory keys and valid BLOCKED/PASSED/SKIPPED labels."""
    cases, _ = load_fixtures()
    valid_labels = {"BLOCKED", "PASSED", "SKIPPED"}
    for i, c in enumerate(cases):
        for key in ("sample_id", "family", "is_attack", "predictions"):
            assert key in c, "Record %d missing %r" % (i, key)
        sid = c.get("sample_id", "index_%d" % i)
        for comp in COMPARATORS:
            assert comp in c["predictions"], (
                "%s missing prediction for %r" % (sid, comp)
            )
            lbl = c["predictions"][comp]
            assert lbl in valid_labels, (
                "%s[%s] invalid label %r (must be one of %s)" % (sid, comp, lbl, valid_labels)
            )


def test_no_skipped_llama_predictions():
    """
    LlamaPromptGuard2 must be BLOCKED or PASSED — not SKIPPED.
    SKIPPED signals that HF_TOKEN was absent and the model was not loaded,
    making the comparator's metrics meaningless.
    """
    cases, _ = load_fixtures()
    skipped = [
        c["sample_id"]
        for c in cases
        if c["predictions"].get("LlamaPromptGuard2") == "SKIPPED"
    ]
    assert len(skipped) == 0, (
        "LlamaPromptGuard2 was SKIPPED for %d cases: %s\n"
        "Model was not loaded (HF_TOKEN missing or load failure).\n"
        "Set HF_TOKEN with access to meta-llama/Llama-Prompt-Guard-2-22M "
        "and re-run the benchmark." % (len(skipped), skipped)
    )


def test_overall_metrics_match_per_case_recomputation():
    """
    CORE REGRESSION TEST: every scalar in 'overall' of security_benchmark_v2.json
    must exactly equal the value recomputed from security_per_case_v2.json.
    """
    cases, bench = load_fixtures()
    computed, _ = recompute_metrics(cases)
    reported    = bench["summary"]["overall"]

    scalar_fields = [
        "TP", "FP", "TN", "FN",
        "accuracy_pct", "precision_pct", "recall_mitigation_pct",
        "f1_score", "false_positive_rate_pct",
    ]

    failures = []
    for comp in COMPARATORS:
        if comp not in reported:
            failures.append("Comparator %r missing from reported overall" % comp)
            continue
        for fld in scalar_fields:
            cv = computed[comp].get(fld)
            rv = reported[comp].get(fld)
            if cv != rv:
                failures.append("[%s] %s: computed=%s, reported=%s" % (comp, fld, cv, rv))

    assert not failures, (
        "\nMetric discrepancies (%d total):\n" % len(failures)
        + "\n".join("  * " + m for m in failures)
    )


def test_per_family_rates_match_per_case_recomputation():
    """
    FAMILY REGRESSION TEST: every family detection rate in 'by_family' must
    exactly match the value recomputed from per-case records.
    """
    cases, bench = load_fixtures()
    _, comp_fam  = recompute_metrics(cases)
    rep_fam      = bench["summary"].get("by_family", {})

    failures = []
    for fam, cd in comp_fam.items():
        if fam not in rep_fam:
            failures.append("Family %r missing from reported by_family" % fam)
            continue
        for comp in COMPARATORS:
            if comp not in cd:
                continue
            for fld in ("total", "detected", "rate_pct"):
                cv = cd[comp].get(fld)
                rv = rep_fam[fam].get(comp, {}).get(fld)
                if cv != rv:
                    failures.append(
                        "[%s][%s] %s: computed=%s, reported=%s" % (fam, comp, fld, cv, rv)
                    )

    assert not failures, (
        "\nFamily metric discrepancies (%d total):\n" % len(failures)
        + "\n".join("  * " + m for m in failures)
    )


def test_combined_dual_regex_union_invariant():
    """
    Combined_Dual_Regex must equal Regex_Only OR Heuristic_Regex_Layer_2
    at per-case level — verify the structural union invariant holds throughout.
    """
    cases, _ = load_fixtures()
    violations = []
    for c in cases:
        p   = c["predictions"]
        r1  = (p.get("Regex_Only") == "BLOCKED")
        r2  = (p.get("Heuristic_Regex_Layer_2") == "BLOCKED")
        comb = (p.get("Combined_Dual_Regex") == "BLOCKED")
        if comb != (r1 or r2):
            violations.append(
                "  %s: Regex_Only=%s, Heuristic_L2=%s => "
                "expected Combined=%s, got %s"
                % (c["sample_id"], r1, r2, r1 or r2, comb)
            )
    assert not violations, (
        "Union invariant violated in %d case(s):\n" % len(violations)
        + "\n".join(violations)
    )


def test_benign_zero_false_positives():
    """
    No benign control (is_attack=False) should be BLOCKED by any comparator.
    FP > 0 would mean the guardrail incorrectly flags legitimate resume text.
    """
    cases, _ = load_fixtures()
    fps = {}
    for c in [case for case in cases if not case["is_attack"]]:
        for comp in COMPARATORS:
            if c["predictions"].get(comp) == "BLOCKED":
                fps.setdefault(comp, []).append(c["sample_id"])
    assert not fps, "False positives on benign controls: " + str(fps)


def test_classifier_type_labels_correct():
    """
    The classifier_type field in every per-case record must match the
    expected implementation type for each comparator.
    """
    cases, _ = load_fixtures()
    expected_types = {
        "Regex_Only":              "deterministic_regex",
        "Heuristic_Regex_Layer_2": "deterministic_regex",
        "Combined_Dual_Regex":     "deterministic_regex_union",
        "LlamaPromptGuard2":       "ml_transformer_inference",
    }
    # Allow 'unavailable' for LlamaPromptGuard2 when model failed to load
    # (test_no_skipped_llama_predictions will separately flag that condition)
    allowed_pg2 = {"ml_transformer_inference", "unavailable"}

    failures = []
    for c in cases:
        ct  = c.get("classifier_types", {})
        sid = c.get("sample_id", "?")
        for comp, exp in expected_types.items():
            actual  = ct.get(comp)
            ok_vals = allowed_pg2 if comp == "LlamaPromptGuard2" else {exp}
            if actual not in ok_vals:
                failures.append(
                    "  %s[%s]: got %r, expected one of %s" % (sid, comp, actual, ok_vals)
                )
    assert not failures, (
        "%d classifier_type issue(s):\n" % len(failures) + "\n".join(failures)
    )


def test_benchmark_structure_integrity():
    """Top-level structure of security_benchmark_v2.json must be intact."""
    _, bench = load_fixtures()
    assert "benchmark" in bench,  "Missing top-level 'benchmark' key"
    assert "summary"   in bench,  "Missing top-level 'summary' key"
    summary = bench["summary"]
    assert "overall"  in summary, "Missing 'summary.overall'"
    assert "by_family" in summary, "Missing 'summary.by_family'"
    assert "llm_guard_integration_status" in summary, (
        "Missing 'summary.llm_guard_integration_status'"
    )
    status = summary["llm_guard_integration_status"]
    for k in ("model_id", "threshold", "integrated", "inference_type"):
        assert k in status, "llm_guard_integration_status missing %r" % k


# ── plain runner (python test_security_benchmark_v2_regression.py) ─────────────

ALL_TESTS = [
    test_fixture_files_exist,
    test_per_case_record_counts,
    test_all_per_case_records_have_required_fields,
    test_no_skipped_llama_predictions,
    test_overall_metrics_match_per_case_recomputation,
    test_per_family_rates_match_per_case_recomputation,
    test_combined_dual_regex_union_invariant,
    test_benign_zero_false_positives,
    test_classifier_type_labels_correct,
    test_benchmark_structure_integrity,
]

if __name__ == "__main__":
    passed = failed = 0
    errs = []
    print("=" * 70)
    print("Security Benchmark v2 -- Regression Tests")
    print("=" * 70)
    for fn in ALL_TESTS:
        try:
            fn()
            print("  [PASS] %s" % fn.__name__)
            passed += 1
        except AssertionError as exc:
            print("  [FAIL] %s" % fn.__name__)
            for ln in str(exc).strip().splitlines():
                print("         %s" % ln)
            errs.append((fn.__name__, str(exc)))
            failed += 1
        except Exception as exc:
            print("  [ERROR] %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))
            errs.append((fn.__name__, "%s: %s" % (type(exc).__name__, exc)))
            failed += 1
    print("=" * 70)
    print("Results: %d passed, %d failed of %d" % (passed, failed, len(ALL_TESTS)))
    if failed == 0:
        print("ALL TESTS PASSED -- aggregate metrics are mathematically consistent "
              "with per-case records.")
    else:
        print("FAILED:")
        for name, msg in errs:
            print("  * %s: %s" % (name, msg[:200]))
    print("=" * 70)
    sys.exit(0 if failed == 0 else 1)
