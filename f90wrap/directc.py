"""Utilities for direct-C code generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

from f90wrap import fortran as ft


INTRINSIC_TYPES = {"integer", "real", "logical", "complex"}


@dataclass(frozen=True)
class ProcedureKey:
    """Identifier for a procedure within a module/type scope."""

    module: str | None
    type_name: str | None
    name: str


@dataclass
class InteropInfo:
    """Interop classification for a procedure."""

    requires_helper: bool


def bind_c_symbol(prefix: str, key: ProcedureKey) -> str:
    """Return the name of the BIND(C) shim for a procedure."""

    parts = [prefix]
    if key.module:
        parts.append(f"{key.module}__")
    if key.type_name:
        parts.append(f"{key.type_name}__")
    parts.append(key.name)
    parts.append("_c")
    return ''.join(parts)


def _argument_is_iso_c(arg: ft.Argument, kind_map: Dict[str, Dict[str, str]]) -> bool:
    """Best-effort test for ISO C compatibility."""

    # Normalize attributes to strings for robust checking
    attrs = [str(a) for a in getattr(arg, "attributes", [])]
    arg_type = str(getattr(arg, "type", ""))

    if any(attr.startswith("intent(out)") for attr in attrs) and arg_type.startswith("character"):
        return False

    if any(attr.startswith("value") for attr in attrs):
        # value arguments are fine when the type is scalar
        pass

    if any(attr.startswith("optional") for attr in attrs):
        return False

    if any(attr.startswith("pointer") or attr.startswith("allocatable")
           for attr in attrs):
        return False

    # Dimension attributes are allowed only for explicit-shape arrays
    dims = [attr for attr in attrs if attr.startswith("dimension")]
    if dims:
        dim_expr = dims[0][len("dimension("):-1]
        if any("*" in part for part in dim_expr.split(",")):
            return False

    ftype = arg_type.strip().lower()
    if ftype.startswith("type(") or ftype.startswith("class("):
        return False

    base, _, kind = ftype.partition("(")
    base = base.strip()
    if base not in INTRINSIC_TYPES:
        return False

    if kind:
        kind = kind.rstrip(") ")
        # Map via kind_map when possible
        if base in kind_map and kind in kind_map[base]:
            c_type = kind_map[base][kind]
            if c_type not in {"int", "float", "double", "long_long"}:
                return False
        else:
            # Unknown kind
            return False

    return True


def _procedure_requires_helper(proc: ft.Procedure, kind_map: Dict[str, Dict[str, str]]) -> bool:
    """Determine if procedure needs a classic f90wrap helper for direct-C mode."""

    # If procedure has any attributes (e.g. recursive), keep helper
    if proc.attributes:
        return True

    for arg in proc.arguments:
        if not _argument_is_iso_c(arg, kind_map):
            return True

    if isinstance(proc, ft.Function):
        if not _argument_is_iso_c(proc.ret_val, kind_map):
            return True

    return False


def analyse_interop(tree: ft.Root, kind_map: Dict[str, Dict[str, str]]) -> Dict[ProcedureKey, InteropInfo]:
    """Analyse the transformed tree and flag which procedures need helpers."""

    classification: Dict[ProcedureKey, InteropInfo] = {}

    def record(procs: Iterable[ft.Procedure]):
        if procs is None:
            return
        try:
            iterator = iter(procs)
        except TypeError:
            return
        for proc in iterator:
            key = ProcedureKey(proc.mod_name, getattr(proc, 'type_name', None), proc.name)
            classification[key] = InteropInfo(
                requires_helper=_procedure_requires_helper(proc, kind_map)
            )

    modules_attr = getattr(tree, "modules", [])
    if modules_attr is None:
        modules_iter = []
    else:
        try:
            modules_iter = list(modules_attr)
        except TypeError:
            modules_iter = []

    for module in modules_iter:
        record(getattr(module, 'procedures', []))

        # Also record procedures inside interfaces (for --keep-single-interface)
        interfaces_attr = getattr(module, 'interfaces', [])
        if interfaces_attr is not None:
            try:
                interfaces_iter = list(interfaces_attr)
            except TypeError:
                interfaces_iter = []
            for iface in interfaces_iter:
                record(getattr(iface, 'procedures', []))

        types_attr = getattr(module, 'types', [])
        if types_attr is None:
            types_iter = []
        else:
            try:
                types_iter = list(types_attr)
            except TypeError:
                types_iter = []
        for derived in types_iter:
            record(getattr(derived, 'procedures', []))

    record(getattr(tree, 'procedures', []))

    return classification


class NamespaceHelper:
    """Utility for scoping and use-statement management in direct-C mode."""

    def __init__(self, types: Dict[str, ft.Type], namespace_types: bool = False):
        self._types = types
        self._namespace_types = namespace_types
        self._modules: Dict[str, ft.Module] = {}

    def register_modules(self, root: ft.Root) -> None:
        """Cache module nodes by both generated and original names."""
        self._modules = {}
        modules = getattr(root, "modules", []) or []
        for module in modules:
            self._modules[module.name] = module
            orig = getattr(module, "orig_name", None)
            if orig:
                self._modules[orig] = module

    def scope_identifier_for(self, container: ft.Fortran) -> str:
        """Build a stable identifier used to namespace generated helper names."""
        if isinstance(container, ft.Module) or not self._namespace_types:
            return container.name
        if isinstance(container, ft.Type):
            owner = getattr(container, "mod_name", None)
            if owner is None:
                owner = self.type_owner(container.name)
            if owner:
                return f"{owner}__{container.name}"
            return container.name
        raise TypeError(f"Unsupported container for scope identifier {container!r}")

    def find_type(self, type_name: str, module_hint: Optional[str] = None) -> Optional[ft.Type]:
        """Locate a Type node, preferring the provided module hint."""
        base = ft.strip_type(type_name)
        search_modules: List[ft.Module] = []
        if module_hint:
            hint = self._modules.get(module_hint)
            if hint and hint not in search_modules:
                search_modules.append(hint)
        for module in self._modules.values():
            if module not in search_modules:
                search_modules.append(module)
        for module in search_modules:
            for typ in getattr(module, "types", []):
                if typ.name == base:
                    return typ
        return self._types.get(base)

    def type_owner(self, type_name: str, module_hint: Optional[str] = None) -> Optional[str]:
        """Return the defining module name for a given type."""
        type_node = self.find_type(type_name, module_hint)
        if type_node is not None:
            return getattr(type_node, "mod_name", None)
        return None

    @staticmethod
    def _ensure_use_entry(extra_uses: Dict[str, Dict[str, object]], module_name: str) -> Dict[str, object]:
        entry = extra_uses.get(module_name)
        if entry is None:
            entry = {"symbols": [], "full": False}
            extra_uses[module_name] = entry
        return entry

    def add_extra_use(self, extra_uses: Dict[str, Dict[str, object]],
                      module_name: Optional[str], symbol: Optional[str]) -> None:
        """Append a symbol to a module's ONLY list, avoiding duplicates."""
        if not module_name:
            return
        entry = self._ensure_use_entry(extra_uses, module_name)
        if symbol is None:
            entry["full"] = True
            return
        if entry["full"] and (not isinstance(symbol, str) or "=>" not in symbol):
            return
        if symbol not in entry["symbols"]:
            entry["symbols"].append(symbol)

    def set_namespace_types(self, enable: bool) -> None:
        """Toggle namespacing behaviour."""
        self._namespace_types = enable
