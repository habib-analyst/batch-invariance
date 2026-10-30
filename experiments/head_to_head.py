"""Head-to-head (PREREG-locked): gated vs ungated pipelines on FRESH seeds.

Master seeds 777000 (tasks) / 777001 (benign run) -- never used in pilot.
Writes results/headline.json (no timestamps; deterministic). Re-running
must produce a byte-identical file (verified by the caller via SHA-256).
"""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from batch_invariance.analyzer import analyze
from batch_invariance.generate import (Candidate, INV_TEMPLATES,
                                       VAR_TEMPLATES, make_rows,
                                       make_task_pool)
from batch_invariance.pipeline import CONTRACT, run_task
from batch_invariance.simulate import check_bitwise

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "headline.json")
N_TASKS = 120
N_BOOT = 2000

def _geomean(xs):
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if len(xs) else 0.0

def main():
    task_seeds = np.random.SeedSequence(777000).spawn(N_TASKS)
    base_pass, gated_pass = [], []
    base_sp, gated_sp = [], []
    disease_num = disease_den = 0
    gated_empty = 0
    # Ablation: per-candidate gate outcomes (paper Table 3 analog).
    static_det = dynamic_det = both_det = missed = 0
    n_variant_cands = 0
    per_task = []
    for i, ss in enumerate(task_seeds):
        rng = np.random.default_rng(ss)
        pool = make_task_pool(rng)
        target, fillers = make_rows(rng)
        res = run_task(pool, target, fillers)
        base_pass.append(res["baseline_pass"])
        gated_pass.append(res["gated_pass"])
        if res["baseline_pick"]:
            base_sp.append(res["baseline_pick"]["speedup"])
        if res["gated_pick"]:
            gated_sp.append(res["gated_pick"]["speedup"])
        else:
            gated_empty += 1
        # Disease + ablation: dynamic behavior is measured INDEPENDENTLY
        # for every variant candidate (pipeline order would truncate
        # dynamic results for static-rejected candidates and understate
        # both the disease rate and the static/dynamic overlap).
        for cand, ev in zip(pool, res["evaluations"]):
            if cand.kind != "variant":
                continue
            n_variant_cands += 1
            dyn = check_bitwise(cand.source, target, fillers)
            if not dyn["pass"]:
                disease_num += 1
            disease_den += 1
            s_reject = ev["static"]["status"] == "REJECT"
            d_fail = not dyn["pass"]
            if s_reject and d_fail:
                both_det += 1
            elif s_reject:
                static_det += 1
            elif d_fail:
                dynamic_det += 1
            else:
                missed += 1
        per_task.append({"task": i,
                         "baseline_pass": bool(res["baseline_pass"]),
                         "gated_pass": bool(res["gated_pass"]),
                         "baseline_speedup": res["baseline_pick"]["speedup"] if res["baseline_pick"] else None,
                         "gated_speedup": res["gated_pick"]["speedup"] if res["gated_pick"] else None})
    base_rate = sum(base_pass) / N_TASKS
    gated_rate = sum(gated_pass) / N_TASKS
    margin_pp = 100 * (gated_rate - base_rate)
    retention = 100 * _geomean(gated_sp) / _geomean(base_sp)
    # Paired bootstrap over tasks.
    rngb = np.random.default_rng(777002)
    idx = np.arange(N_TASKS)
    b_arr = np.array(base_pass, dtype=float)
    g_arr = np.array(gated_pass, dtype=float)
    b_sp = np.array([t["baseline_speedup"] for t in per_task], dtype=float)
    g_sp = np.array([t["gated_speedup"] for t in per_task], dtype=float)
    margins, retentions = [], []
    for _ in range(N_BOOT):
        s = rngb.integers(0, N_TASKS, size=N_TASKS)
        margins.append(100 * (g_arr[s].mean() - b_arr[s].mean()))
        ok = ~np.isnan(b_sp[s]) & ~np.isnan(g_sp[s])
        if ok.sum() >= 10:
            retentions.append(100 * _geomean(g_sp[s][ok]) / _geomean(b_sp[s][ok]))
    def ci(xs):
        lo, hi = np.percentile(xs, [2.5, 97.5])
        return [round(float(lo), 2), round(float(hi), 2)]
    # Benign run: invariant-only pools, fresh stream 777001.
    benign_seeds = np.random.SeedSequence(777001).spawn(N_TASKS)
    rejected = total = 0
    for ss in benign_seeds:
        rng = np.random.default_rng(ss)
        inv_names = list(INV_TEMPLATES)
        for _ in range(3):
            t = inv_names[int(rng.integers(0, len(inv_names)))]
            total += 1
            res = run_task([Candidate(INV_TEMPLATES[t], "invariant", t, 20.0)],
                           *make_rows(rng))
            if not res["gated_pass"] or res["gated_empty"]:
                rejected += 1
    headline = {
        "n_tasks": N_TASKS,
        "baseline_pass_rate": round(base_rate, 4),
        "gated_pass_rate": round(gated_rate, 4),
        "margin_pp": round(margin_pp, 2),
        "margin_ci95": ci(margins),
        "geomean_baseline_speedup": round(_geomean(base_sp), 4),
        "geomean_gated_speedup": round(_geomean(gated_sp), 4),
        "speedup_retention_pct": round(retention, 2),
        "retention_ci95": ci(retentions),
        "disease_rate": round(disease_num / disease_den, 4) if disease_den else 0.0,
        "disease_numerator": disease_num,
        "disease_denominator": disease_den,
        "gated_empty_tasks": gated_empty,
        "ablation_candidates": {
            "n_variant_candidates": n_variant_cands,
            "static_only": static_det, "dynamic_only": dynamic_det,
            "both": both_det, "missed_by_both": missed,
        },
        "benign_false_rejection_rate": round(rejected / total, 4) if total else 0.0,
        "benign_rejected": rejected, "benign_total": total,
        "gates": {
            "G2_disease_ge_80": (disease_num / disease_den) >= 0.80 if disease_den else False,
            "G3a_margin_ge_15pp": margin_pp >= 15,
            "G3a_ci_lower_gt_0": ci(margins)[0] > 0,
            "G3b_retention_ge_65": retention >= 65,
            "G3c_false_rejection_le_10": (rejected / total) <= 0.10 if total else False,
        },
    }
    headline["all_gates_pass"] = all(headline["gates"].values())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(headline, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(headline, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
