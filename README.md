# batch-invariance — CPU analog of BIEvolve (arXiv 2610.36623)

**Scope honesty, first:** this is a CPU-only analog of Bao et al.'s
BIEvolve, not a reproduction. Real GPU kernels cannot run on this VM, so
no number here is a kernel measurement. The paper's figures (19/83
confirmed violations, BI pass 67.6% → 91.9%, 91.8% speedup retention) are
context only. What this repo demonstrates, with measured numbers, is the
*mechanism*: contract-guided static dependency analysis + dynamic
bitwise-consistency validation across batch contexts, applied to
simulated reduction kernels where batch context changes reduction
partitioning — the exact failure mode the paper names (floating-point
non-associativity under changed reduction order).

## The phenomenon (measured, CPU)
A float32 reduction split into k chunks gives different output *bits*
for different k. Across batch contexts {1, 2, 4, 8, 16} (target row last
in every context), a batch-dependent split count therefore makes the
same input produce different bits depending on its batchmates.

## The gates (mirroring the paper, sec. 3)
1. **Static** (`batch_invariance/analyzer.py`): parses the kernel source
   to an AST and propagates batch-context taint from contract-declared
   batch symbols (plus context projections `len(batch)`,
   `batch.shape[0]`) to protected parameters — through assignments,
   aliases (resolved via the contract), helper functions, container
   accesses, and batch-dependent conditionals/loops. The contract itself
   is a parsed DSL (`contracts.py`), not a blocklist: batch symbols,
   protected parameters, policies (`fixed_across_batch` /
   `may_depend_on batch` / `free`), aliases, and `extends` inheritance
   are all declared in contract text and resolved before analysis.
   Verdicts: PASS / REJECT / UNKNOWN (dynamic construction such as
   `exec` over batch-derived code cannot be resolved statically, so it
   is routed to dynamic validation — neither auto-passed nor
   auto-rejected, per the paper).
2. **Correctness** (separate from BI, as in the paper): float64 reference
   reduction, rtol 1e-3. Tolerance agreement does not imply BI.
3. **Dynamic** (`simulate.py`): executes the candidate source on the
   target row embedded in each batch context and requires bitwise-
   identical target outputs.
4. **Selection**: fastest surviving candidate by (generator-drawn)
   speedup proxy. The ungated baseline applies correctness only.

## Results (head-to-head, 120 tasks, fresh seeds — see `PREREG.md`)
- Disease: 209/240 variant candidates (87.1%) show a cross-context
  bitwise mismatch (bar ≥ 80%). The 31 misses are the threshold class,
  whose split threshold sits above every tested context — dynamically
  invisible by design, caught by the static gate.
- Gated BI pass 100% vs ungated baseline 35%: margin **+65.0pp**,
  paired-bootstrap 95% CI [55.8, 73.3] (bars: ≥ +15pp, CI lower > 0).
- Speedup retention (geomean gated/ungated): **79.0%**, CI [74.9, 83.1]
  (bar ≥ 65%). Gating costs speed — the tradeoff is reported, not hidden.
- Benign bound: 0/360 genuinely invariant candidates falsely rejected
  (bar ≤ 10%).
- Ablation (240 variant candidates): static-only 31, dynamic-only 27,
  both 182, missed by both 0 — the two gates are complementary, as in
  the paper's Table 3.
- Determinism: `results/headline.json` re-runs byte-identical
  (SHA-256 `9a39d14307b892222827d542a4a4e4d9eeb08f4d4dc66746363db2e9188ee627`).

## Run
```bash
python3 -m pytest tests/ -q          # 11 tests
python3 experiments/pilot.py         # pilot-only seeds -> results/pilot.json
python3 experiments/head_to_head.py  # PREREG-locked seeds -> results/headline.json
```

## Layout
- `batch_invariance/` — `contracts.py` (DSL), `analyzer.py` (static),
  `simulate.py` (dynamic + correctness), `generate.py` (candidates),
  `pipeline.py` (gated vs ungated selection)
- `experiments/` — `pilot.py`, `head_to_head.py`
- `tests/` — 11 tests
- `results/` — `pilot.json`, `headline.json`
- `PREREG.md` (locked before head-to-head), `KILLGATE.md` (gate record)

## Limits (honest)
- Simulated reductions, not GPU kernels; no tiling, no attention, no
  MoE routing, no real agent loop (candidates are template-generated,
  speedup is a drawn proxy conditioned on kind, not a benchmark).
- The dynamic context set is finite ({1, 2, 4, 8, 16}); a threshold
  above 16 is invisible to it — which is exactly why the static gate
  exists, and why the disease bar is 80% rather than 95%.
- The static analyzer covers Python sources; CUDA/C++ would need the
  paper's pattern-based path, not built here.

## Attribution
Idea and mechanism: Bao, Wang & Chen, "God Does Not Play Dice:
Enforcing Batch Invariance in Agentic Kernel Generation" (arXiv
2610.36623, STAR Lab, George Mason / NUS, submitted 27 Oct 2026).
This is an independent CPU analog built from the paper's description;
no author code was available (verified 2026-10-30: no code link on the
abs page; full-text HTML github matches are cited repos only).
