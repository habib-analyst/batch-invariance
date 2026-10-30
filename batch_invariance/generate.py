"""Deterministic candidate generation for the CPU analog.

A candidate is a kernel source text plus a generator-drawn speedup proxy.
The speedup proxy plays the role of the paper's benchmarked speedup: it is
drawn from a kind-conditioned distribution (aggressive, batch-dependent
tiling is faster on average -- the paper's Fig. 5 observation that gating
excludes "fast but batch-dependent optimization choices") and is NEVER
visible to either gate. The gates see only source text (static) and
executed behavior (dynamic). Ground-truth kind labels are used ONLY to
score pipelines, never to decide them.

Pool composition per task (locked in PREREG.md): 6 candidates -- 3
invariant, 2 variant, 1 incorrect (passes static, fails tolerance
correctness, so the correctness stage does real work). Templates are
chosen by seeded RNG; variable names for protected parameters vary across
the contract's alias set so the static gate must resolve aliases.
"""

import numpy as np

CONTRACT_TEXT = """
kernel reduction {
  invariant_axis batch symbols [batch_size, batch];
  parameter num_splits {
    semantic split_structure;
    policy fixed_across_batch;
    aliases [n_splits, splits];
  }
  parameter tile_size {
    semantic split_structure;
    policy fixed_across_batch;
    aliases [tile];
  }
  parameter rng_seed {
    semantic rng_stream;
    policy fixed_across_batch;
    aliases [seed];
  }
  parameter out_shape {
    semantic split_structure;
    policy may_depend_on batch;
    aliases [shape];
  }
}
kernel attention extends reduction {
  parameter num_kv_splits {
    semantic split_structure;
    policy fixed_across_batch;
    aliases [];
  }
}
"""

def _body(k_expr_lines, extra=""):
    """Assemble a kernel source: k computed by k_expr_lines, then reduce."""
    lines = ["def kernel(batch, batch_size):",
             "    outs = []",
             "    for row in batch:"]
    for ln in k_expr_lines:
        lines.append("        " + ln)
    if extra:
        for ln in extra.split("\n"):
            lines.append("        " + ln)
    lines.append("        outs.append(split_reduce(row, num_splits))")
    lines.append("    return np.array(outs, dtype=np.float32)")
    return "\n".join(lines) + "\n"

# Invariant templates: protected parameter never batch-derived.
INV_TEMPLATES = {
    "inv_fixed": _body(["num_splits = 4"]),
    "inv_alias": _body(["n_splits = 4"]).replace("num_splits))", "n_splits))"),
    "inv_benign_shape": _body(["num_splits = 4"],
                              extra="out_shape = (batch_size,)\n"
                                    "n_rows = len(batch)"),
    "inv_tile_const": _body(["tile = 64", "num_splits = 1024 // tile"]),
}
# Fix inv_alias: the reduce line must reference the alias name.
INV_TEMPLATES["inv_alias"] = (
    "def kernel(batch, batch_size):\n"
    "    outs = []\n"
    "    for row in batch:\n"
    "        n_splits = 4\n"
    "        outs.append(split_reduce(row, n_splits))\n"
    "    return np.array(outs, dtype=np.float32)\n"
)

# Variant templates: one mechanism each (paper Table A1 / Fig. 2).
VAR_TEMPLATES = {
    # Work partitioning, transitive through two assignments (Fig. 2 style).
    "var_partition": _body(["work = batch_size * 8",
                            "num_splits = work // 8"]),
    # Tiling: tile size derived from batch size drives the split count.
    "var_tile": _body(["tile = batch_size * 16",
                       "num_splits = 1024 // tile"]),
    # Algorithm selection through a batch-dependent conditional.
    "var_algo_select": _body(["if batch_size > 4:",
                              "    num_splits = 8",
                              "else:",
                              "    num_splits = 2"]),
    # Threshold above every dynamic context (static-unique catch: the
    # dynamic contexts {1,2,4,8,16} never cross 32).
    "var_threshold": _body(["if batch_size > 32:",
                            "    num_splits = 16",
                            "else:",
                            "    num_splits = 4"]),
    # Batch dependence through a same-file helper function.
    "var_helper": (
        "def calc_splits(b):\n"
        "    return b * 2\n"
        "def kernel(batch, batch_size):\n"
        "    outs = []\n"
        "    for row in batch:\n"
        "        num_splits = calc_splits(batch_size)\n"
        "        outs.append(split_reduce(row, num_splits))\n"
        "    return np.array(outs, dtype=np.float32)\n"
    ),
    # Batch-dependent loop bound selecting the split count.
    "var_loop": _body(["num_splits = 2",
                       "for i in range(batch_size):",
                       "    num_splits = num_splits + 1"]),
    # Batch-derived shape projection.
    "var_shape": _body(["num_splits = batch.shape[0] * 2"]),
    # Dynamic construction: static analysis returns UNKNOWN, so only the
    # dynamic gate can catch this (dynamic-unique class).
    "var_exec": _body(["ns = {}",
                       "exec(\"k = \" + str(batch_size * 2), ns)",
                       "num_splits = ns[\"k\"]"]),
}

# Incorrect template: statically clean, fixed splits, but output scaled so
# the tolerance correctness stage (not any BI gate) rejects it.
INCORRECT_TEMPLATE = (
    "def kernel(batch, batch_size):\n"
    "    outs = []\n"
    "    for row in batch:\n"
    "        num_splits = 4\n"
    "        outs.append(split_reduce(row, num_splits))\n"
    "    return np.array(outs, dtype=np.float32) * np.float32(1.5)\n"
)

# Speedup-proxy lognormal medians (locked in PREREG.md); sigma shared.
SPEEDUP_MEDIAN = {"invariant": 20.0, "variant": 26.0, "incorrect": 30.0}
SPEEDUP_SIGMA = 0.35

class Candidate:
    def __init__(self, source, kind, template, speedup):
        self.source = source
        self.kind = kind  # "invariant" | "variant" | "incorrect"
        self.template = template
        self.speedup = float(speedup)

    def to_dict(self):
        return {"kind": self.kind, "template": self.template,
                "speedup": round(self.speedup, 4)}

def make_task_pool(rng):
    """6 candidates: 3 invariant, 2 variant, 1 incorrect (fixed mix)."""
    pool = []
    inv_names = list(INV_TEMPLATES)
    var_names = list(VAR_TEMPLATES)
    for _ in range(3):
        t = inv_names[int(rng.integers(0, len(inv_names)))]
        pool.append(Candidate(INV_TEMPLATES[t], "invariant", t,
                              rng.lognormal(np.log(SPEEDUP_MEDIAN["invariant"]),
                                            SPEEDUP_SIGMA)))
    for _ in range(2):
        t = var_names[int(rng.integers(0, len(var_names)))]
        pool.append(Candidate(VAR_TEMPLATES[t], "variant", t,
                              rng.lognormal(np.log(SPEEDUP_MEDIAN["variant"]),
                                            SPEEDUP_SIGMA)))
    pool.append(Candidate(INCORRECT_TEMPLATE, "incorrect", "incorrect",
                          rng.lognormal(np.log(SPEEDUP_MEDIAN["incorrect"]),
                                        SPEEDUP_SIGMA)))
    order = rng.permutation(len(pool))
    return [pool[i] for i in order]

def make_rows(rng, n=1024, kind="mixed"):
    """One target row + filler rows for a task (fixed per task by seed)."""
    if kind == "mixed":
        def gen():
            mag = np.exp(rng.uniform(np.log(1e-2), np.log(1e3), size=n))
            sign = rng.choice(np.array([-1.0, 1.0]), size=n)
            return (mag * sign).astype(np.float32)
    elif kind == "normal":
        def gen():
            return rng.standard_normal(n).astype(np.float32)
    elif kind == "uniform":
        def gen():
            return rng.random(n).astype(np.float32)
    else:
        raise ValueError(kind)
    target = gen()
    fillers = [gen() for _ in range(15)]  # enough for B=16
    return target, fillers
