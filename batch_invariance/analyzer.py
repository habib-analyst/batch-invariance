"""Contract-guided static dependency analysis over kernel source text.

Implements the static module of Bao et al. (2610.36623) sec. 3.1 as an
AST taint analysis for Python kernel sources:

- Sources: the contract's batch symbols, plus context projections of the
  batch array (``len(batch)``, ``batch.shape[0]``). Plain data uses of the
  batch array (row iteration, row values) are NOT sources.
- Sinks: assignments to protected parameters (contract
  ``fixed_across_batch`` parameters, resolved through aliases).
- Propagation: data dependence through assignments, augmented assignment,
  container accesses, and same-file helper functions (argument taint bound
  to parameters, return taint propagated to the caller); control dependence
  through ``if``/``while``/``for`` conditions and iteration expressions
  (an assignment inside a batch-dependent branch or loop is tainted even if
  its right-hand side is clean).
- Verdicts: PASS (no protected parameter tainted), REJECT (a protected
  parameter is tainted, with the violating line and source names), or
  UNKNOWN (the source uses dynamic construction -- ``eval``/``exec`` or a
  batch-derived dynamic attribute -- that static analysis cannot resolve;
  per the paper, such candidates are left to dynamic validation, NOT
  auto-rejected and NOT auto-passed).

Deliberately conservative choices are documented where they occur: an
unknown function call propagates the union of its argument taints, and a
non-constant index into ``batch.shape`` is treated as context.
"""

import ast

BATCH_ARRAY = "batch"  # harness batch-array parameter name (see simulate.py)

class Verdict:
    def __init__(self, status, violations=(), reason=""):
        assert status in ("PASS", "REJECT", "UNKNOWN")
        self.status = status
        self.violations = list(violations)  # [{parameter, line, sources}]
        self.reason = reason

    def to_dict(self):
        return {"status": self.status, "violations": self.violations,
                "reason": self.reason}

def _is_const(node, value):
    return isinstance(node, ast.Constant) and node.value == value

class _Analyzer:
    def __init__(self, tree, contract):
        self.tree = tree
        self.contract = contract
        self.batch_symbols = set(contract.batch_symbols)
        self.alias_map = contract.alias_map
        self.protected = contract.protected()
        self.functions = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self.functions[node.name] = node
        self.violations = []
        self.unknown_reason = ""
        self._call_stack = ()

    # ---- expression taint ----
    def taint(self, node, env, ctrl):
        """Taint set of an expression; env maps names to taint sets."""
        if node is None:
            return set()
        if isinstance(node, ast.Constant):
            return set()
        if isinstance(node, ast.Name):
            if node.id in env:
                # Assigned names (and parameters, pre-populated) shadow the
                # declared-symbol default, so reassignment is respected.
                return set(env[node.id])
            if node.id == BATCH_ARRAY:
                return set()  # the batch array as data is not context
            if node.id in self.batch_symbols:
                return {node.id}
            return set()
        if isinstance(node, ast.BinOp):
            return self.taint(node.left, env, ctrl) | self.taint(node.right, env, ctrl)
        if isinstance(node, ast.UnaryOp):
            return self.taint(node.operand, env, ctrl)
        if isinstance(node, ast.BoolOp):
            out = set()
            for v in node.values:
                out |= self.taint(v, env, ctrl)
            return out
        if isinstance(node, ast.Compare):
            out = self.taint(node.left, env, ctrl)
            for c in node.comparators:
                out |= self.taint(c, env, ctrl)
            return out
        if isinstance(node, ast.IfExp):
            return (self.taint(node.test, env, ctrl)
                    | self.taint(node.body, env, ctrl)
                    | self.taint(node.orelse, env, ctrl))
        if isinstance(node, ast.Call):
            # len(batch) / batch.shape context projections
            if isinstance(node.func, ast.Name) and node.func.id == "len" and node.args:
                if isinstance(node.args[0], ast.Name) and node.args[0].id in self.batch_symbols:
                    return {node.args[0].id}
            if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                if node.args and (self.taint(node.args[0], env, ctrl) & self.batch_symbols):
                    self.unknown_reason = f"dynamic {node.func.id} over batch-derived code (line {node.lineno})"
                return set()
            if isinstance(node.func, ast.Name) and node.func.id in self.functions:
                return self._call_helper(node, env, ctrl)
            out = set()
            for a in node.args:
                out |= self.taint(a, env, ctrl)
            for kw in node.keywords:
                out |= self.taint(kw.value, env, ctrl)
            return out
        if isinstance(node, ast.Attribute):
            # batch.shape[0]-style projections are handled in Subscript;
            # bare batch.shape is context; anything else propagates the value.
            if (isinstance(node.value, ast.Name)
                    and node.value.id in self.batch_symbols and node.attr == "shape"):
                return set(self.batch_symbols & {node.value.id})
            return self.taint(node.value, env, ctrl)
        if isinstance(node, ast.Subscript):
            # batch.shape[0] -> context; batch.shape[k] non-const -> context
            # (conservative); batch.shape[1] -> feature dim, not context.
            v = node.value
            if (isinstance(v, ast.Attribute) and v.attr == "shape"
                    and isinstance(v.value, ast.Name)
                    and v.value.id in self.batch_symbols):
                sl = node.slice
                if _is_const(sl, 0):
                    return {v.value.id}
                if isinstance(sl, ast.Constant):
                    return set()
                return {v.value.id} | self.taint(sl, env, ctrl)
            return self.taint(v, env, ctrl) | self.taint(node.slice, env, ctrl)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            out = set()
            for e in node.elts:
                out |= self.taint(e, env, ctrl)
            return out
        if isinstance(node, ast.Dict):
            out = set()
            for k, v in zip(node.keys, node.values):
                if k is not None:
                    out |= self.taint(k, env, ctrl)
                out |= self.taint(v, env, ctrl)
            return out
        if isinstance(node, ast.Starred):
            return self.taint(node.value, env, ctrl)
        if isinstance(node, ast.ListComp):
            # Generators: element taint + iterator taint (conservative for
            # comprehensions over batch-derived ranges).
            out = self.taint(node.elt, env, ctrl)
            for gen in node.generators:
                out |= self.taint(gen.iter, env, ctrl)
            return out
        # Fallback: union over child expressions.
        out = set()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                out |= self.taint(child, env, ctrl)
        return out

    def _call_helper(self, node, env, ctrl):
        name = node.func.id
        if name in self._call_stack:
            return set()  # recursion guard: unresolvable, treated as clean
        func = self.functions[name]
        if len(node.args) != len(func.args.args):
            # Arity mismatch: fall back to union of argument taints.
            out = set()
            for a in node.args:
                out |= self.taint(a, env, ctrl)
            return out
        sub = dict(env)
        for param, arg in zip(func.args.args, node.args):
            sub[param.arg] = self.taint(arg, env, ctrl)
        self._call_stack = self._call_stack + (name,)
        try:
            ret = self._exec_block(func.body, sub, set(ctrl))
        finally:
            self._call_stack = self._call_stack[:-1]
        return ret

    # ---- statements ----
    def _assign_target(self, target, taint, env, lineno):
        if isinstance(target, ast.Name):
            env[target.id] = set(taint)
            canon = self.alias_map.get(target.id)
            if canon in self.protected and (set(taint) & self.batch_symbols):
                self.violations.append({
                    "parameter": canon,
                    "assigned_as": target.id,
                    "line": lineno,
                    "sources": sorted(set(taint) & self.batch_symbols),
                })
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._assign_target(elt, taint, env, lineno)
        # Subscript/attribute targets: container writes do not create a
        # protected-parameter binding; their index/value taint is irrelevant
        # to the sink set (protected parameters are plain names).

    def _exec_block(self, stmts, env, ctrl):
        """Execute statements; return taint of any returned expression."""
        ret = set()
        for stmt in stmts:
            r = self._exec(stmt, env, ctrl)
            if r is not None:
                ret |= r
        return ret

    def _exec(self, stmt, env, ctrl):
        if isinstance(stmt, ast.Assign):
            t = self.taint(stmt.value, env, ctrl) | set(ctrl)
            for tgt in stmt.targets:
                self._assign_target(tgt, t, env, stmt.lineno)
        elif isinstance(stmt, ast.AugAssign):
            old = set(env.get(stmt.target.id, ())) if isinstance(stmt.target, ast.Name) else set()
            t = old | self.taint(stmt.value, env, ctrl) | set(ctrl)
            self._assign_target(stmt.target, t, env, stmt.lineno)
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            t = self.taint(stmt.value, env, ctrl) | set(ctrl)
            self._assign_target(stmt.target, t, env, stmt.lineno)
        elif isinstance(stmt, ast.If):
            ct = self.taint(stmt.test, env, ctrl)
            self._exec_block(stmt.body, env, ctrl | ct)
            self._exec_block(stmt.orelse, env, ctrl | ct)
        elif isinstance(stmt, ast.While):
            ct = self.taint(stmt.test, env, ctrl)
            self._exec_block(stmt.body, env, ctrl | ct)
            self._exec_block(stmt.orelse, env, ctrl | ct)
        elif isinstance(stmt, ast.For):
            it = self.taint(stmt.iter, env, ctrl)
            # Loop variable binds the iteration value: for data iteration
            # (rows of batch) this is data (untainted); for range(batch_size)
            # the taint above already captures context.
            if isinstance(stmt.target, ast.Name):
                env[stmt.target.id] = set(it)
            self._exec_block(stmt.body, env, ctrl | it)
            self._exec_block(stmt.orelse, env, ctrl | it)
        elif isinstance(stmt, ast.Return):
            return self.taint(stmt.value, env, ctrl) | set(ctrl)
        elif isinstance(stmt, ast.Expr):
            self.taint(stmt.value, env, ctrl)
        elif isinstance(stmt, ast.FunctionDef):
            pass  # helpers resolved on demand
        else:
            # Other statements (import, etc.): walk child expressions for
            # dynamic-construct detection.
            for child in ast.walk(stmt):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) \
                        and child.func.id in ("eval", "exec"):
                    self.unknown_reason = f"dynamic {child.func.id} (line {child.lineno})"
        return None

def analyze(source, contract, entry="kernel"):
    """Analyze kernel source text against a contract. Returns Verdict."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return Verdict("UNKNOWN", reason=f"source does not parse: {e}")
    an = _Analyzer(tree, contract)
    func = an.functions.get(entry)
    if func is None:
        return Verdict("UNKNOWN", reason=f"no function {entry!r} in source")
    env = {}
    for arg in func.args.args:
        # Parameters named as batch symbols are batch sources; the batch
        # array itself is data (its projections are handled in taint()).
        env[arg.arg] = {arg.arg} if arg.arg in an.batch_symbols and arg.arg != BATCH_ARRAY else set()
    an._exec_block(func.body, env, set())
    if an.unknown_reason:
        # Dynamic construction: static analysis cannot decide. Route to
        # dynamic validation (paper sec. 3.1), do not auto-reject.
        return Verdict("UNKNOWN", reason=an.unknown_reason)
    if an.violations:
        return Verdict("REJECT", violations=an.violations,
                       reason="protected parameter depends on batch context")
    return Verdict("PASS", reason="no protected parameter depends on batch context")
