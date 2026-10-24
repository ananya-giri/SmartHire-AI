"""
Audit script: recompute all metrics from security_per_case_v2.json
and compare against security_benchmark_v2.json.
"""
import json, math

PER_CASE = "data/security_per_case_v2.json"
BENCHMARK = "data/security_benchmark_v2.json"

with open(PER_CASE, "r", encoding="utf-8") as f:
    cases = json.load(f)

with open(BENCHMARK, "r", encoding="utf-8") as f:
    bench = json.load(f)

print(f"Total per-case records: {len(cases)}")

# ── Count by family & label ──────────────────────────────────────────────────
families = {}
for c in cases:
    fam = c["family"]
    families.setdefault(fam, {"attack": 0, "benign": 0})
    if c["is_attack"]:
        families[fam]["attack"] += 1
    else:
        families[fam]["benign"] += 1

attack_total = sum(v["attack"] for v in families.values())
benign_total = sum(v["benign"] for v in families.values())
print(f"Attack cases: {attack_total}, Benign cases: {benign_total}")
print("\nFamily breakdown:")
for fam in sorted(families):
    d = families[fam]
    print(f"  {fam}: attack={d['attack']}, benign={d['benign']}")

# ── Recompute confusion matrices ─────────────────────────────────────────────
COMPARATORS = ["Regex_Only", "Heuristic_Regex_Layer_2", "Combined_Dual_Regex", "LlamaPromptGuard2"]
metrics = {c: {"TP":0,"FP":0,"TN":0,"FN":0} for c in COMPARATORS}
family_breakdown = {}

for c in cases:
    is_atk = c["is_attack"]
    fam = c["family"]
    family_breakdown.setdefault(fam, {comp: {"total":0,"detected":0} for comp in COMPARATORS})
    for comp in COMPARATORS:
        pred = c["predictions"].get(comp, "PASSED")
        blocked = (pred == "BLOCKED")
        if is_atk and blocked:       metrics[comp]["TP"] += 1
        elif not is_atk and blocked: metrics[comp]["FP"] += 1
        elif not is_atk and not blocked: metrics[comp]["TN"] += 1
        elif is_atk and not blocked: metrics[comp]["FN"] += 1
        # family
        family_breakdown[fam][comp]["total"] += 1
        if blocked:
            family_breakdown[fam][comp]["detected"] += 1

print("\n\n=== RECOMPUTED vs REPORTED (Overall) ===")
overall_bench = bench["summary"]["overall"]
discrepancies = []

for comp in COMPARATORS:
    m = metrics[comp]
    tp, fp, tn, fn = m["TP"], m["FP"], m["TN"], m["FN"]
    total = tp + fp + tn + fn
    acc  = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0

    reported = overall_bench.get(comp, {})
    r_tp  = reported.get("TP", "?")
    r_fp  = reported.get("FP", "?")
    r_tn  = reported.get("TN", "?")
    r_fn  = reported.get("FN", "?")
    r_acc = reported.get("accuracy_pct", "?")
    r_rec = reported.get("recall_mitigation_pct", "?")
    r_prec = reported.get("precision_pct", "?")
    r_f1  = reported.get("f1_score", "?")
    r_fpr = reported.get("false_positive_rate_pct", "?")

    print(f"\n  [{comp}]")
    print(f"    TP  : computed={tp}, reported={r_tp}  {'OK' if tp==r_tp else 'MISMATCH!'}")
    print(f"    FP  : computed={fp}, reported={r_fp}  {'OK' if fp==r_fp else 'MISMATCH!'}")
    print(f"    TN  : computed={tn}, reported={r_tn}  {'OK' if tn==r_tn else 'MISMATCH!'}")
    print(f"    FN  : computed={fn}, reported={r_fn}  {'OK' if fn==r_fn else 'MISMATCH!'}")
    print(f"    Acc : computed={round(acc*100,2)}, reported={r_acc}  {'OK' if round(acc*100,2)==r_acc else 'MISMATCH!'}")
    print(f"    Prec: computed={round(prec*100,2)}, reported={r_prec}  {'OK' if round(prec*100,2)==r_prec else 'MISMATCH!'}")
    print(f"    Rec : computed={round(rec*100,2)}, reported={r_rec}  {'OK' if round(rec*100,2)==r_rec else 'MISMATCH!'}")
    print(f"    F1  : computed={round(f1,4)}, reported={r_f1}  {'OK' if round(f1,4)==r_f1 else 'MISMATCH!'}")
    print(f"    FPR : computed={round(fpr*100,2)}, reported={r_fpr}  {'OK' if round(fpr*100,2)==r_fpr else 'MISMATCH!'}")

    if tp != r_tp or fp != r_fp or tn != r_tn or fn != r_fn:
        discrepancies.append((comp, "confusion matrix"))

print("\n\n=== RECOMPUTED vs REPORTED (Per-Family Detection Rates) ===")
bench_fam = bench["summary"]["by_family"]
family_discrepancies = []

for fam in sorted(family_breakdown):
    print(f"\n  Family: {fam}")
    bench_fam_data = bench_fam.get(fam, {})
    for comp in COMPARATORS:
        d = family_breakdown[fam][comp]
        computed_total = d["total"]
        computed_detected = d["detected"]
        computed_rate = round((computed_detected/computed_total)*100, 1) if computed_total > 0 else 0.0
        reported_fam = bench_fam_data.get(comp, {})
        r_total = reported_fam.get("total", "?")
        r_detected = reported_fam.get("detected", "?")
        r_rate = reported_fam.get("rate_pct", "?")
        status = "OK" if (computed_total==r_total and computed_detected==r_detected and computed_rate==r_rate) else "MISMATCH!"
        if status == "MISMATCH!":
            family_discrepancies.append((fam, comp, computed_total, r_total, computed_detected, r_detected, computed_rate, r_rate))
        print(f"    [{comp}] total={computed_total}(rep={r_total}), detected={computed_detected}(rep={r_detected}), rate={computed_rate}%(rep={r_rate}%)  {status}")

print("\n\n=== SUMMARY ===")
print(f"Overall discrepancies: {discrepancies if discrepancies else 'NONE'}")
print(f"Family discrepancies: {family_discrepancies if family_discrepancies else 'NONE'}")

# ── Special investigation: Combined_Dual_Regex TP=23 vs per-case ─────────────
print("\n\n=== INVESTIGATION: Combined_Dual_Regex detections ===")
cdr_blocked = [(c["sample_id"], c["family"]) for c in cases if c["is_attack"] and c["predictions"]["Combined_Dual_Regex"]=="BLOCKED"]
cdr_missed  = [(c["sample_id"], c["family"]) for c in cases if c["is_attack"] and c["predictions"]["Combined_Dual_Regex"]=="PASSED"]
print(f"BLOCKED (TP): {len(cdr_blocked)} -> {cdr_blocked}")
print(f"PASSED  (FN): {len(cdr_missed)} -> {cdr_missed}")

# ── Per-case LlamaGuard labels ───────────────────────────────────────────────
print("\n\n=== LlamaPromptGuard2 per-case labels ===")
for c in cases:
    if c["is_attack"]:
        print(f"  {c['sample_id']} ({c['family'][:20]:20s}): {c['predictions']['LlamaPromptGuard2']}")
