"""BI contract DSL: a genuinely specified contract language, not a blocklist.

Concrete syntax (mirrors the BNF in Bao et al. 2610.36623, Fig. 4)::

    kernel <id> [extends <id>, ...] {
      invariant_axis batch symbols [<name>, ...];
      parameter <name> {
        semantic <split_structure|recurrence_structure|rng_stream>;
        policy <fixed_across_batch|may_depend_on batch|free>;
        aliases [<name>, ...];
      }
      ...
    }

Semantics (locked before any head-to-head data):
- ``invariant_axis batch symbols`` declares the batch-context variables.
  A source-level name is a batch source iff it is one of these symbols, or a
  *context projection* of the batch array (``len(batch)``, ``batch.shape[0]``)
  where ``batch`` itself is a declared symbol. Plain uses of the batch array
  as data (iterating its rows, indexing a row) are NOT batch sources: data
  dependence on batch contents is not batch-context dependence.
- ``policy fixed_across_batch``: the parameter must never be assigned from
  (or selected by) a batch source, directly or transitively. This is the
  load-bearing protection; ``may_depend_on batch`` and ``free`` parameters
  are explicitly unprotected, so a gate that rejected them would be wrong.
- ``aliases``: alternate source-level names denoting the same parameter.
  The analyzer resolves aliases through the contract, so renaming a
  protected parameter does not evade the gate.
- ``extends``: inheritance is resolved before analysis; a child inherits the
  parent's batch symbols and parameters, overriding parameters by name.

The parser is a small tokenizer + recursive-descent parser. Any syntax
error, unknown semantic/policy keyword, or duplicate parameter is a hard
error, never a silent pass.
"""

import re

SEMANTICS = {"split_structure", "recurrence_structure", "rng_stream"}
POLICIES = {"fixed_across_batch", "may_depend_on batch", "free"}

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[{}[\],;]")

class ContractError(ValueError):
    pass

class Parameter:
    def __init__(self, name, semantic, policy, aliases):
        self.name = name
        self.semantic = semantic
        self.policy = policy
        self.aliases = list(aliases)

    def to_dict(self):
        return {"name": self.name, "semantic": self.semantic,
                "policy": self.policy, "aliases": self.aliases}

class Contract:
    def __init__(self, name, batch_symbols, parameters):
        self.name = name
        self.batch_symbols = list(batch_symbols)
        self.parameters = list(parameters)  # list[Parameter]
        # alias -> canonical parameter name (canonical name maps to itself)
        self.alias_map = {}
        for p in self.parameters:
            if p.name in self.alias_map:
                raise ContractError(f"duplicate parameter {p.name!r}")
            self.alias_map[p.name] = p.name
            for a in p.aliases:
                if a in self.alias_map and self.alias_map[a] != p.name:
                    raise ContractError(f"alias {a!r} claimed by two parameters")
                self.alias_map[a] = p.name

    def protected(self):
        """Canonical names of fixed_across_batch parameters."""
        return {p.name for p in self.parameters
                if p.policy == "fixed_across_batch"}

    def to_dict(self):
        return {"name": self.name, "batch_symbols": self.batch_symbols,
                "parameters": [p.to_dict() for p in self.parameters]}

def _tokenize(text):
    toks = _TOKEN_RE.findall(text)
    # Reject any non-whitespace character that is not part of a token.
    stripped = _TOKEN_RE.sub("", text)
    if stripped.strip():
        raise ContractError(f"unexpected characters in contract: {stripped.strip()[:40]!r}")
    return toks

class _Parser:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        t = self.peek()
        if t is None:
            raise ContractError("unexpected end of contract")
        self.i += 1
        return t

    def expect(self, t):
        got = self.next()
        if got != t:
            raise ContractError(f"expected {t!r}, got {got!r}")

    def name(self):
        t = self.next()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t):
            raise ContractError(f"expected identifier, got {t!r}")
        return t

    def name_list(self):
        self.expect("[")
        out = []
        if self.peek() != "]":
            out.append(self.name())
            while self.peek() == ",":
                self.next()
                out.append(self.name())
        self.expect("]")
        return out

def parse_contracts(text):
    """Parse all kernel blocks; resolve inheritance; return {name: Contract}."""
    p = _Parser(_tokenize(text))
    raw = {}  # name -> (parents, batch_symbols, [Parameter])
    order = []
    while p.peek() is not None:
        if p.next() != "kernel":
            raise ContractError("expected 'kernel'")
        name = p.name()
        parents = []
        if p.peek() == "extends":
            p.next()
            parents.append(p.name())
            while p.peek() == ",":
                p.next()
                parents.append(p.name())
        p.expect("{")
        batch_symbols = None
        params = []
        while p.peek() != "}":
            kw = p.next()
            if kw == "invariant_axis":
                if p.next() != "batch":
                    raise ContractError("expected 'batch' after invariant_axis")
                if p.next() != "symbols":
                    raise ContractError("expected 'symbols'")
                if batch_symbols is not None:
                    raise ContractError("duplicate invariant_axis block")
                batch_symbols = p.name_list()
                p.expect(";")
            elif kw == "parameter":
                pname = p.name()
                p.expect("{")
                semantic = policy = None
                aliases = []
                while p.peek() != "}":
                    f = p.next()
                    if f == "semantic":
                        semantic = p.next()
                        if semantic not in SEMANTICS:
                            raise ContractError(f"unknown semantic {semantic!r}")
                        p.expect(";")
                    elif f == "policy":
                        first = p.next()
                        if first == "may_depend_on":
                            if p.next() != "batch":
                                raise ContractError("expected 'batch' after may_depend_on")
                            policy = "may_depend_on batch"
                        elif first in ("fixed_across_batch", "free"):
                            policy = first
                        else:
                            raise ContractError(f"unknown policy {first!r}")
                        p.expect(";")
                    elif f == "aliases":
                        aliases = p.name_list()
                        p.expect(";")
                    else:
                        raise ContractError(f"unknown parameter field {f!r}")
                p.expect("}")
                if semantic is None or policy is None:
                    raise ContractError(f"parameter {pname!r} missing semantic/policy")
                params.append(Parameter(pname, semantic, policy, aliases))
            else:
                raise ContractError(f"unknown block keyword {kw!r}")
        p.expect("}")
        if name in raw:
            raise ContractError(f"duplicate kernel {name!r}")
        # A child may inherit invariant_axis from its parents; a kernel
        # with neither its own block nor a parent block fails at resolve.
        raw[name] = (parents, batch_symbols, params)
        order.append(name)
    # Resolve inheritance (parents must be defined; child overrides by name).
    resolved = {}
    def resolve(name, stack=()):
        if name in resolved:
            return resolved[name]
        if name in stack:
            raise ContractError(f"inheritance cycle at {name!r}")
        parents, symbols, params = raw[name]
        merged_symbols = list(symbols) if symbols is not None else []
        merged = {}
        for par in parents:
            if par not in raw:
                raise ContractError(f"kernel {name!r} extends unknown {par!r}")
            pc = resolve(par, stack + (name,))
            for s in pc.batch_symbols:
                if s not in merged_symbols:
                    merged_symbols.append(s)
            for pp in pc.parameters:
                merged[pp.name] = pp
        if not merged_symbols:
            raise ContractError(f"kernel {name!r} declares no invariant_axis")
        for pp in params:
            merged[pp.name] = pp
        c = Contract(name, merged_symbols, list(merged.values()))
        resolved[name] = c
        return c
    for name in order:
        resolve(name)
    return resolved
