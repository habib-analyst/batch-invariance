"""Tests for batch-invariance (run: python3 -m pytest tests/ -q, or unittest)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from batch_invariance.analyzer import analyze
from batch_invariance.contracts import parse_contracts, ContractError
from batch_invariance.generate import (CONTRACT_TEXT, INV_TEMPLATES,
                                       VAR_TEMPLATES, INCORRECT_TEMPLATE,
                                       make_task_pool, make_rows)
from batch_invariance.pipeline import CONTRACT, run_task
from batch_invariance.simulate import (check_bitwise, check_correctness,
                                       split_reduce)

CONTRACTS = parse_contracts(CONTRACT_TEXT)

def test_dsl_parses_and_inherits():
    att = CONTRACTS["attention"]
    assert att.batch_symbols == ["batch_size", "batch"]
    names = {p.name for p in att.parameters}
    assert "num_splits" in names and "num_kv_splits" in names
    assert att.alias_map["n_splits"] == "num_splits"

def test_dsl_rejects_unknown_policy():
    bad = ("kernel k { invariant_axis batch symbols [batch_size];"
           " parameter p { semantic split_structure; policy bogus; } }")
    try:
        parse_contracts(bad)
    except ContractError:
        return
    raise AssertionError("unknown policy accepted")

def test_dsl_rejects_duplicate_alias():
    bad = ("kernel k { invariant_axis batch symbols [batch_size];"
           " parameter a { semantic split_structure; policy free; aliases [x]; }"
           " parameter b { semantic split_structure; policy free; aliases [x]; } }")
    try:
        parse_contracts(bad)
    except ContractError:
        return
    raise AssertionError("duplicate alias accepted")

def test_static_detects_all_variant_mechanisms():
    for name, src in VAR_TEMPLATES.items():
        v = analyze(src, CONTRACT)
        if name == "var_exec":
            assert v.status == "UNKNOWN", (name, v.to_dict())
        else:
            assert v.status == "REJECT", (name, v.to_dict())

def test_static_passes_all_invariant_templates():
    for name, src in INV_TEMPLATES.items():
        v = analyze(src, CONTRACT)
        assert v.status == "PASS", (name, v.to_dict())
    assert analyze(INCORRECT_TEMPLATE, CONTRACT).status == "PASS"

def test_static_control_and_alias():
    # Control dependence: constant value selected by a batch condition.
    src = ("def kernel(batch, batch_size):\n"
           "    outs = []\n"
           "    for row in batch:\n"
           "        if batch_size > 1:\n"
           "            n_splits = 4\n"
           "        else:\n"
           "            n_splits = 4\n"
           "        outs.append(split_reduce(row, n_splits))\n"
           "    return np.array(outs, dtype=np.float32)\n")
    v = analyze(src, CONTRACT)
    assert v.status == "REJECT" and v.violations[0]["parameter"] == "num_splits"

def test_split_reduce_diverges_with_partition_count():
    rng = np.random.default_rng(0)
    row = (np.exp(rng.uniform(np.log(1e-2), np.log(1e3), size=1024))
           * rng.choice(np.array([-1.0, 1.0]), size=1024)).astype(np.float32)
    bits = {split_reduce(row, k).tobytes() for k in (1, 2, 4, 8, 16)}
    assert len(bits) > 1

def test_dynamic_invariant_passes_variant_fails():
    rng = np.random.default_rng(7)
    target, fillers = make_rows(rng)
    assert check_bitwise(INV_TEMPLATES["inv_fixed"], target, fillers)["pass"]
    assert not check_bitwise(VAR_TEMPLATES["var_partition"], target, fillers)["pass"]
    # Dynamic misses the above-threshold variant (contexts never cross 32).
    assert check_bitwise(VAR_TEMPLATES["var_threshold"], target, fillers)["pass"]
    # ...while static catches it: the complementarity the paper reports.
    assert analyze(VAR_TEMPLATES["var_threshold"], CONTRACT).status == "REJECT"
    # ...and dynamic catches what static cannot resolve.
    assert not check_bitwise(VAR_TEMPLATES["var_exec"], target, fillers)["pass"]

def test_correctness_separates_incorrect():
    rng = np.random.default_rng(11)
    target, fillers = make_rows(rng)
    batch = np.asarray([target] + fillers[:3], dtype=np.float32)
    assert check_correctness(INV_TEMPLATES["inv_fixed"], batch)["pass"]
    assert not check_correctness(INCORRECT_TEMPLATE, batch)["pass"]

def test_pool_determinism():
    a = make_task_pool(np.random.default_rng(42))
    b = make_task_pool(np.random.default_rng(42))
    assert [(c.kind, c.template, c.speedup) for c in a] == \
           [(c.kind, c.template, c.speedup) for c in b]

def test_gated_task_picks_invariant():
    rng = np.random.default_rng(99)
    pool = make_task_pool(rng)
    target, fillers = make_rows(rng)
    res = run_task(pool, target, fillers)
    assert res["gated_pick"] is not None
    assert res["gated_pick"]["kind"] == "invariant"
