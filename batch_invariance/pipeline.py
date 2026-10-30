"""Gated vs ungated candidate selection (the paper's sec. 3.3 pipeline).

Order (paper's four stages): (1) static BI analysis, (2) reference
correctness test, (3) dynamic BI validation, (4) speedup selection among
survivors. A static UNKNOWN verdict does NOT reject: the candidate
proceeds to dynamic validation, which decides. A static REJECT skips all
later stages. The ungated baseline applies only stage (2) correctness,
then picks the fastest candidate -- exactly the "correctness and speed
only" regime the paper's baseline agents represent.

A task's BI pass is scored against ground-truth kind: the picked
candidate must be genuinely invariant. (An incorrect candidate can never
pass: correctness filters it in both pipelines.)
"""

import numpy as np

from .analyzer import analyze
from .contracts import parse_contracts
from .generate import CONTRACT_TEXT
from .simulate import check_bitwise, check_correctness

CONTRACTS = parse_contracts(CONTRACT_TEXT)
CONTRACT = CONTRACTS["reduction"]

def evaluate_candidate(cand, target, fillers, batch_for_correctness):
    """Run all gates on one candidate; return a per-stage outcome dict."""
    static = analyze(cand.source, CONTRACT)
    out = {"kind": cand.kind, "template": cand.template,
           "speedup": round(cand.speedup, 4),
           "static": static.to_dict(), "correct": None, "dynamic": None,
           "survived": False}
    # Correctness is evaluated for every candidate: the ungated baseline
    # needs it independently of the static gate (the baseline pipeline does
    # not run static analysis at all). Gating still blocks REJECTs below.
    corr = check_correctness(cand.source, batch_for_correctness)
    out["correct"] = corr
    if static.status == "REJECT":
        return out
    if not corr["pass"]:
        return out
    dyn = check_bitwise(cand.source, target, fillers)
    out["dynamic"] = dyn
    if not dyn["pass"]:
        return out
    out["survived"] = True
    return out

def _pick_max_speedup(cands):
    if not cands:
        return None
    return max(cands, key=lambda c: c.speedup)

def run_task(pool, target, fillers):
    """Run both pipelines on one task pool; return per-pipeline results."""
    batch_for_correctness = np.asarray([target] + fillers[:3], dtype=np.float32)
    evals = [evaluate_candidate(c, target, fillers, batch_for_correctness)
             for c in pool]
    # Baseline: correctness only, then fastest.
    base_eligible = [c for c, e in zip(pool, evals)
                     if e["correct"] is not None and e["correct"]["pass"]]
    base_pick = _pick_max_speedup(base_eligible)
    # Gated: full pipeline, then fastest survivor.
    gated_eligible = [c for c, e in zip(pool, evals) if e["survived"]]
    gated_pick = _pick_max_speedup(gated_eligible)
    return {
        "baseline_pick": base_pick.to_dict() if base_pick else None,
        "baseline_pass": bool(base_pick and base_pick.kind == "invariant"),
        "gated_pick": gated_pick.to_dict() if gated_pick else None,
        "gated_pass": bool(gated_pick and gated_pick.kind == "invariant"),
        "gated_empty": gated_pick is None,
        "evaluations": evals,
    }
