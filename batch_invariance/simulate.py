"""Dynamic bitwise-consistency validation across batch contexts (CPU analog).

The paper's dynamic module (2610.36623 sec. 3.2) executes a candidate on a
target sample embedded in different batch contexts and requires bitwise-
identical target outputs. Real GPU kernels cannot run on this VM, so the
analog executes the candidate's own Python source (a reduction kernel over
a batch of row vectors) with numpy float32 arithmetic -- the exact
mechanism the paper names (floating-point non-associativity under changed
reduction partitioning, sec. 2.1) -- and compares target-output bits.

Rules, locked before head-to-head:
- Contexts: batch sizes {1, 2, 4, 8, 16}; the target row sits at position 0
  for B=1 and at the last position otherwise (position varies, as in the
  paper's context set). Filler rows are fixed per (task, input) by seed.
- Correctness (separate from BI, as in the paper): the kernel's output on
  a reference batch must match a float64 reference reduction within
  rtol=1e-3. Tolerance agreement does NOT imply BI, and BI failure is not
  a correctness failure.
- BI verdict: PASS iff the target row's float32 bits are identical across
  all contexts; otherwise FAIL with the first mismatching context.
- Execution is sandboxed to a namespace exposing only numpy and the
  helpers the kernel sources use (``split_reduce``); no file, network, or
  subprocess access is provided, and the source is pre-screened for
  ``import`` statements (any import -> execution refused, verdict ERROR).
"""

import ast
import numpy as np

CONTEXTS = (1, 2, 4, 8, 16)
RTOL = 1e-3

def split_reduce(row, k):
    """float32 reduction of a row split into k contiguous chunks.

    Each chunk is summed sequentially in float32, then partials are summed
    in order -- so the result's bits depend on k (non-associativity). This
    is the CPU analog of a GPU split-reduction whose partition count is
    set by the kernel's tiling logic.
    """
    n = len(row)
    partials = []
    for i in range(k):
        s, e = i * n // k, (i + 1) * n // k
        acc = np.float32(0.0)
        for v in row[s:e]:
            acc = np.float32(acc + v)
        partials.append(acc)
    acc = np.float32(0.0)
    for p in partials:
        acc = np.float32(acc + p)
    return acc

def _load(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("kernel source must not import modules")
    ns = {"np": np, "split_reduce": split_reduce}
    exec(compile(tree, "<kernel>", "exec"), ns)
    if "kernel" not in ns or not callable(ns["kernel"]):
        raise ValueError("kernel source must define kernel(batch, batch_size)")
    return ns["kernel"]

def run_kernel(source, batch):
    """Run candidate source on a float32 batch array; return float32 outputs."""
    fn = _load(source)
    out = fn(np.asarray(batch, dtype=np.float32), int(batch.shape[0]))
    arr = np.asarray(out, dtype=np.float32).reshape(-1)
    if arr.shape[0] != batch.shape[0]:
        raise ValueError("kernel output length must equal batch size")
    return arr

def reference_reduce(row):
    return float(np.sum(np.asarray(row, dtype=np.float64)))

def check_correctness(source, batch):
    """Tolerance-based correctness vs float64 reference (independent of BI)."""
    try:
        out = run_kernel(source, batch)
    except Exception as e:
        return {"pass": False, "reason": f"execution error: {e}"}
    for i, row in enumerate(batch):
        ref = reference_reduce(row)
        if abs(float(out[i]) - ref) > RTOL * max(1.0, abs(ref)):
            return {"pass": False,
                    "reason": f"row {i} output {float(out[i])} vs reference {ref}"}
    return {"pass": True, "reason": ""}

def check_bitwise(source, target, filler_rows):
    """Bitwise consistency of the target row across CONTEXTS.

    filler_rows: list of row vectors; context B uses the first B-1 of them.
    Returns dict(pass, baseline_bits, mismatches=[{context, bits}]).
    """
    try:
        baseline = None
        mismatches = []
        for B in CONTEXTS:
            # Target row last in every context (position 0 for B=1, last
            # otherwise), so its position varies with the batch context.
            rows = list(filler_rows[:B - 1]) + [target]
            batch = np.asarray(rows, dtype=np.float32)
            out = run_kernel(source, batch)
            pos = 0 if B == 1 else B - 1
            bits = np.asarray(out[pos], dtype=np.float32).tobytes().hex()
            if baseline is None:
                baseline = bits
            elif bits != baseline:
                mismatches.append({"context": B, "bits": bits})
        return {"pass": not mismatches, "baseline_bits": baseline,
                "mismatches": mismatches}
    except Exception as e:
        return {"pass": False, "baseline_bits": None,
                "mismatches": [], "error": str(e)}
