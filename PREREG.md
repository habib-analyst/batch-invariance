# PREREG — project #99 `batch-invariance` (LOCKED 2026-10-30, before head-to-head)

**Decision rule (standing, #97 lesson): ANY locked bar fails → KILL.
No post-hoc bar changes, no bar-shrinking, no amendments, no re-running
with different seeds to "fix" a failure. A misspecified bar kills the
project honestly.**

## Context (not results)
Paper: arXiv 2610.36623, Bao et al., "God Does Not Play Dice: Enforcing
Batch Invariance in Agentic Kernel Generation" (BIEvolve). The paper's
figures (19/83 violations, 67.6% → 91.9% pass, 91.8% speedup retention)
are CONTEXT ONLY from real kernel agents — never claimed as reproduced.
This build is a CPU analog only: numpy float32 reduction-order simulation
+ a contract-DSL static analyzer over kernel source text. No GPU kernels
run on this VM, and no number below is presented as a kernel measurement.

## Gates already passed (before this lock)
- G0: deterministic, CPU-only, $0, no LLMs/APIs/GPUs; 11/11 tests green.
- G1 (2026-10-30, live recheck): arXiv abs page has no code link;
  full-text HTML has exactly 2 "github" matches, both cited repos
  (FlashInfer agent baseline [12]; Thinking Machines batch-invariant ops
  [35]) — no BIEvolve author code; web search finds no BIEvolve repo.
  PASS.
- Pilot (pilot-only seeds, base 1000; results/pilot.json): divergence
  200/200 mixed, 200/200 normal, 199/200 uniform; analyzer REJECTs all 7
  statically-visible variant mechanisms, returns UNKNOWN (by design) for
  the exec-constructed variant, and PASSes all 4 invariant templates
  including the benign batch-shape case; pilot head-to-head (30 tasks):
  baseline 30%, gated 100%, margin 70pp, retention 75.0%.

## Head-to-head protocol (locked)
- Tasks: N=120. Master seed 777000; per-task RNG from
  `SeedSequence(777000).spawn(120)`; each task draws its 6-candidate pool
  (fixed mix: 3 invariant, 2 variant, 1 incorrect; templates by RNG;
  speedup lognormal medians invariant 20 / variant 26 / incorrect 30,
  sigma 0.35) then its target + filler rows, in that fixed order.
- Dynamic contexts: batch sizes {1, 2, 4, 8, 16}; target row last in every
  context. Correctness: float64 reference, rtol 1e-3 (separate from BI).
- Benign run (separate stream): N=120 invariant-only pools (3 invariant
  candidates each), master seed 777001, same per-task spawn scheme.
- Statistics: paired bootstrap over tasks, B=2000 resamples, percentile
  95% CIs for (a) the pass-rate margin and (b) the geomean speedup ratio.
  Reported alongside point estimates; the CI bar below applies to (a).
- Determinism: the head-to-head script is re-run and
  results/headline.json must be byte-identical (SHA-256 recorded).

## Locked bars
- **G2 disease:** variant candidates exhibit a cross-context bitwise
  mismatch in **≥ 80%** of (variant candidate, task input) pairs. The bar
  is 80%, not 95%, deliberately: the threshold variant class (split
  threshold above every tested context) is dynamically invisible BY
  DESIGN — it is the static gate's unique catch — and makes up ~1/8 of
  variant draws, so a bar above ~87% would penalize the design itself.
- **G3(a) mechanism, pass-rate margin:** gated BI pass rate minus
  baseline BI pass rate **≥ +15pp**, AND the paired-bootstrap 95% CI
  lower bound **> 0pp**. Both must hold.
- **G3(b) mechanism, speedup tradeoff:** geomean gated speedup /
  geomean ungated speedup (over tasks where both pipelines pick) **≥ 65%**.
  Pilot retention was 75.0%; 65% is the honest analog bar (the analog's
  variant speedup median exceeds the invariant median by construction,
  so analog retention is structurally below the paper's 91.8% — the
  tradeoff is reported, not hidden).
- **G3(c) benign/utility bound:** full-pipeline false rejection of
  genuinely invariant candidates **≤ 10%** (static REJECT or dynamic FAIL
  on the invariant-only run). Paper's analog figure is 2%; 10% is the
  conservative analog bar.
- **Determinism:** headline re-run byte-identical, else FAIL (counts as
  a failed bar → KILL).

## What counts as what (locked definitions)
- BI pass (task): the pipeline's picked candidate (max speedup among its
  survivors) has ground-truth kind "invariant". Incorrect candidates can
  never pass (correctness filters them in both pipelines).
- Gated empty (no survivor): counts as a task failure for the gated arm
  and is reported; it cannot occur for invariant-containing pools unless
  a gate misfires, which the benign bar would also catch.
- Static UNKNOWN is NOT a rejection: the candidate proceeds to dynamic
  validation, which decides. Counting UNKNOWN as rejection would inflate
  the benign rejection rate and misstate the paper's design.
