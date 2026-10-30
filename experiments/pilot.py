"""Pilot (scope gate): proves, on pilot-only seeds, that
(a) bitwise divergence across batch contexts is exhibitable on CPU, and
(b) the contract DSL + analyzer genuinely distinguish violation
    mechanisms from benign batch uses.
Writes results/pilot.json (no timestamps; deterministic).
Pilot seeds (base 1000) are NEVER reused in the head-to-head run.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from batch_invariance.analyzer import analyze
from batch_invariance.generate import (INV_TEMPLATES, VAR_TEMPLATES,
                                       make_rows, make_task_pool)
from batch_invariance.pipeline import CONTRACT, run_task
from batch_invariance.simulate import check_bitwise

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "pilot.json")

def divergence_rates():
    out = {}
    for kind in ("mixed", "normal", "uniform"):
        rng = np.random.default_rng(1000)
        div = 0
        n = 200
        for _ in range(n):
            target, fillers = make_rows(rng, kind=kind)
            if not check_bitwise(VAR_TEMPLATES["var_partition"], target, fillers)["pass"]:
                div += 1
        out[kind] = {"divergent": div, "trials": n, "rate": div / n}
    return out

def analyzer_corpus():
    table = {}
    for name, src in VAR_TEMPLATES.items():
        table[name] = analyze(src, CONTRACT).status
    for name, src in INV_TEMPLATES.items():
        table[name] = analyze(src, CONTRACT).status
    return table

def pilot_head_to_head(n_tasks=30):
    rng = np.random.default_rng(1000)
    base_pass = gated_pass = 0
    base_sp, gated_sp = [], []
    for _ in range(n_tasks):
        pool = make_task_pool(rng)
        target, fillers = make_rows(rng)
        res = run_task(pool, target, fillers)
        base_pass += res["baseline_pass"]
        gated_pass += res["gated_pass"]
        if res["baseline_pick"]:
            base_sp.append(res["baseline_pick"]["speedup"])
        if res["gated_pick"]:
            gated_sp.append(res["gated_pick"]["speedup"])
    import math
    def geomean(xs):
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0
    return {
        "tasks": n_tasks,
        "baseline_pass_rate": base_pass / n_tasks,
        "gated_pass_rate": gated_pass / n_tasks,
        "margin_pp": 100 * (gated_pass - base_pass) / n_tasks,
        "speedup_retention_pct": 100 * geomean(gated_sp) / geomean(base_sp),
    }

def main():
    results = {
        "pilot_seeds": "base 1000 (never reused in head-to-head)",
        "divergence_rates_variant_partition": divergence_rates(),
        "analyzer_corpus_verdicts": analyzer_corpus(),
        "pilot_head_to_head": pilot_head_to_head(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(results, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
