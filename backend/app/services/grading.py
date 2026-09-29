"""Rulebook: per-onion bucket (A / URS / Reject) and the lot decision. All limits come from grading_rules.json."""
import json
import math


def load_rules(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def estimate_weight_g(diameter_mm, density):
    if not diameter_mm:
        return None
    d_cm = diameter_mm / 10
    return density * math.pi / 6 * d_cm ** 3


def grade_onion(defects, diameter_mm, rules):
    """defects: list of defect class names found on this onion. diameter_mm: float or None (not measured).
    Returns (bucket, reasons)."""
    reasons = []
    bad = [d for d in defects if d in rules["reject_defects"]]
    if bad:
        return "Reject", [f"{d}" for d in sorted(set(bad))]

    A, U = rules["grade_a"], rules["urs"]
    urs_defects = [d for d in defects if d in U["allowed_defects"]]

    if diameter_mm is None:
        size_ok_a, size_ok_urs = True, True
        reasons.append("size not measured")
    else:
        size_ok_a = A["min_mm"] <= diameter_mm <= A["max_mm"]
        size_ok_urs = U["min_mm"] <= diameter_mm <= U["max_mm"]
        if diameter_mm < U["min_mm"]:
            return "Reject", [f"undersized ({diameter_mm:.0f} mm < {U['min_mm']} mm)"]
        if diameter_mm > U["max_mm"]:
            return "Reject", [f"oversized ({diameter_mm:.0f} mm > {U['max_mm']} mm)"]

    if size_ok_a and not urs_defects:
        return "A", reasons

    # falls to URS: either size is only within URS range, or has a URS-allowed defect
    if not size_ok_a:
        reasons.append(f"size {diameter_mm:.0f} mm outside Grade A {A['min_mm']}-{A['max_mm']} mm")
    reasons += urs_defects
    if not rules.get("urs_open", True):
        return "Reject", reasons + ["URS buying closed"]
    return "URS", reasons


def lot_summary(onions, rules):
    """onions: list of dicts with 'bucket' and 'weight_g'. Percentages by count and by estimated weight."""
    n = len(onions)
    buckets = ["A", "URS", "Reject"]
    count = {b: sum(1 for o in onions if o["bucket"] == b) for b in buckets}
    pct = {b: (100 * count[b] / n if n else 0.0) for b in buckets}
    w_known = [o for o in onions if o.get("weight_g")]
    w_tot = sum(o["weight_g"] for o in w_known)
    pct_w = {b: (100 * sum(o["weight_g"] for o in w_known if o["bucket"] == b) / w_tot if w_tot else None) for b in buckets}

    L = rules["lot"]
    if n == 0:
        decision = "No onions found"
    elif pct["A"] >= L["accept_a_min_pct"] and pct["Reject"] <= L["max_reject_pct"]:
        decision = "Accept as Grade A"
    elif rules.get("urs_open", True) and pct["A"] + pct["URS"] >= L["accept_urs_min_pct"]:
        decision = "Accept as URS"
    else:
        decision = "Reject / re-sort"

    return {
        "onion_count": n,
        "count": count,
        "pct_by_count": {b: round(v, 1) for b, v in pct.items()},
        "pct_by_weight": {b: (round(v, 1) if v is not None else None) for b, v in pct_w.items()},
        "est_total_weight_g": round(w_tot) if w_tot else None,
        "decision": decision,
        "rules_version": rules["version"],
    }
