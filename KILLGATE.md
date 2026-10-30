# KILLGATE — project #99 `batch-invariance` — ALL GATES PASS (no kill)

Kill-gate-first build, PREREG locked 2026-10-30 after pilot, before any
head-to-head data. Zero PREREG amendments. Decision rule honored: ANY
locked bar fails → KILL; none failed.

| Gate | Verdict | Measured evidence |
|---|---|---|
| G0 harness | PASS | Deterministic, CPU-only, $0, no LLMs/APIs/GPUs; 11/11 tests green |
| G1 author code | PASS | Live recheck 2026-10-30: abs page no code link; full-text HTML exactly 2 github matches, both cited repos (FlashInfer baseline, TML batch-invariant ops); web search: no BIEvolve repo |
| Pilot (scope) | PASS | Divergence 200/200 mixed, 200/200 normal, 199/200 uniform; analyzer REJECTs all 7 statically-visible mechanisms, UNKNOWN for exec variant, PASSes all 4 invariant templates incl. benign batch-shape case; pilot head-to-head margin 70pp, retention 75.0% |
| G2 disease | PASS | 209/240 = 87.08% ≥ 80% bar (31 misses = threshold class, dynamically invisible by design) |
| G3(a) margin | PASS | +65.0pp ≥ 15pp; paired-bootstrap 95% CI [55.83, 73.33], lower > 0 |
| G3(b) retention | PASS | 79.02% ≥ 65%; CI [74.93, 83.13] |
| G3(c) benign | PASS | 0/360 = 0.00% ≤ 10% |
| Determinism | PASS | Headline re-run byte-identical; SHA-256 9a39d14307b892222827d542a4a4e4d9eeb08f4d4dc66746363db2e9188ee627 |

Headline: baseline BI pass 35% → gated 100%; gated-empty tasks 0;
ablation over 240 variant candidates: static-only 31, dynamic-only 27,
both 182, missed 0.

No bar was moved, shrunk, or amended at any point (#97 lesson). The
disease bar (80%, not 95%) and retention bar (65%, not the paper's
91.8%) were set in PREREG with written justification *before* the
head-to-head run, accounting for the designed threshold class and the
analog's kind-conditioned speedup proxy.
