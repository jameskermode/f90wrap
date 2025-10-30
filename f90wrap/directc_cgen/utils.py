"""Utility functions and data classes for Direct-C code generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from f90wrap import fortran as ft
from f90wrap.transform import shorten_long_name
from f90wrap.numpy_utils import c_type_from_fortran


@dataclass
class ModuleHelper:
    """Metadata for module-level helper routines."""

    module: str
    name: str
    kind: str  # 'get', 'set', 'array', 'array_getitem', 'array_setitem', 'array_len', 'get_derived', 'set_derived'
    element: ft.Element
    is_type_member: bool = False


def helper_name(proc: ft.Procedure, prefix: str) -> str:
    """Return the bare f90wrap helper name for a procedure."""
    parts: List[str] = [prefix]
    if proc.mod_name:
        parts.append(f"{proc.mod_name}__")
    parts.append(proc.name)
    return shorten_long_name("".join(parts))


def helper_param_list(proc: ft.Procedure, kind_map: Dict[str, Dict[str, str]]) -> List[str]:
    """Build the C parameter list for a helper declaration."""
    params: List[str] = []
    char_lens: List[str] = []  # Hidden lengths come after all explicit args

    for arg in proc.arguments:
        if is_hidden_argument(arg):
            params.append("int* " + arg.name)
        elif is_derived_type(arg):
            params.append(f"int* {arg.name}")
        elif is_array(arg):
            c_type = c_type_from_fortran(arg.type, kind_map)
            params.append(f"{c_type}* {arg.name}")
            # For character arrays, add element length as hidden argument
            if arg.type.lower().startswith("character"):
                char_lens.append(f"int {arg.name}_elem_len")
        elif arg.type.lower().startswith("character"):
            params.append(f"char* {arg.name}")
            # Save length for later - Fortran puts hidden lengths AFTER all explicit args
            char_lens.append(f"int {arg.name}_len")
        else:
            c_type = c_type_from_fortran(arg.type, kind_map)
            params.append(f"{c_type}* {arg.name}")

    if isinstance(proc, ft.Function):
        c_type = c_type_from_fortran(proc.ret_val.type, kind_map)
        params.insert(0, f"{c_type}* result")

    # Add hidden character lengths at the end (Fortran calling convention)
    params.extend(char_lens)

    return params


def helper_symbol(proc: ft.Procedure, prefix: str) -> str:
    """Return helper name with C macro for symbol mangling."""
    return f"F90WRAP_F_SYMBOL({helper_name(proc, prefix)})"


def module_helper_name(helper: ModuleHelper, prefix: str) -> str:
    """Return helper name for module-level helper routines."""
    base = f"{prefix}{helper.module}__"
    if helper.kind in {"get", "set", "get_derived", "set_derived"}:
        kind_label = "get" if helper.kind.startswith("get") else "set"
        base += f"{kind_label}__{helper.name}"
    elif helper.kind == "array":
        base += f"array__{helper.name}"
    elif helper.kind == "array_getitem":
        base += f"array_getitem__{helper.name}"
    elif helper.kind == "array_setitem":
        base += f"array_setitem__{helper.name}"
    elif helper.kind == "array_len":
        base += f"array_len__{helper.name}"
    return shorten_long_name(base)


def module_helper_symbol(helper: ModuleHelper, prefix: str) -> str:
    """Return C symbol macro for module helper."""
    return f"F90WRAP_F_SYMBOL({module_helper_name(helper, prefix)})"


def character_length_expr(type_spec: str) -> Optional[str]:
    """Extract a fixed character length literal when available."""
    text = type_spec.strip().lower()
    if not text.startswith("character"):
        return None

    if "(" not in text or ")" not in text:
        return None

    inside = text[text.find("(") + 1 : text.rfind(")")].strip()
    if inside.startswith("len="):
        inside = inside[4:].strip()
    if inside == "*" or not inside:
        return None
    inside = inside.replace(" ", "")
    if re.fullmatch(r"\d+", inside):
        return inside
    return None


def static_array_shape(arg: ft.Argument) -> Optional[Tuple[int, ...]]:
    """Return constant shape for dimension(...) attribute when available."""
    for attr in arg.attributes:
        if attr.startswith("dimension(") and attr.endswith(")"):
            entries = [entry.strip() for entry in attr[len("dimension("):-1].split(",")]
            dims: List[int] = []
            for entry in entries:
                if not entry or not entry.isdigit():
                    return None
                dims.append(int(entry))
            return tuple(dims)
    return None


def module_helper_wrapper_name(helper: ModuleHelper) -> str:
    """Return wrapper function name for a module helper.

    Uses '_helper_' prefix to avoid name collisions with procedure wrappers.
    For example, a module variable setter won't collide with a subroutine
    that has a similar name.
    """
    return f"wrap_{helper.module}_helper_{helper.kind}_{helper.name}"


def wrapper_name(module_label: str, proc: ft.Procedure) -> str:
    """Return wrapper function name for a procedure."""
    binding = getattr(proc, "binding", None)
    if binding and getattr(binding, "name", None):
        func_name = binding.name
    else:
        func_name = proc.name

    if getattr(proc, "mod_name", None):
        return f"wrap_{proc.mod_name}_{func_name}"
    return f"wrap_{module_label}_{func_name}"


def is_array(arg: ft.Argument) -> bool:
    """Check if argument is an array."""
    return any("dimension" in attr for attr in arg.attributes)


def is_derived_type(arg: ft.Argument) -> bool:
    """Return True if argument represents a derived type handle."""
    ftype = arg.type.strip().lower()
    return ftype.startswith("type(") or ftype.startswith("class(")


def derived_pointer_name(name: str) -> str:
    """Return a safe C identifier for a derived-type handle argument."""
    return f"{name}_handle" if name == "self" else name


def is_hidden_argument(arg: ft.Argument) -> bool:
    """Return True if argument is hidden from the Python API."""
    return any(attr.startswith("intent(hide)") for attr in arg.attributes)


def arg_intent(arg: ft.Argument) -> str:
    """Return the declared intent for an argument."""
    for attr in arg.attributes:
        if attr.startswith("intent(") and attr.endswith(")"):
            return attr[len("intent(") : -1].strip().lower()
    return "in"


def is_optional(arg: ft.Argument) -> bool:
    """Return True if the argument is optional."""
    return any(attr.strip().lower() == "optional" for attr in arg.attributes)


def should_parse_argument(arg: ft.Argument) -> bool:
    """Determine whether the argument should be parsed from Python."""
    if is_hidden_argument(arg):
        return False

    if is_optional(arg):
        return True

    intent = arg_intent(arg)
    return intent != "out"


def is_output_argument(arg: ft.Argument) -> bool:
    """Return True if the argument contributes to the Python return value."""
    if is_hidden_argument(arg):
        return False

    intent = arg_intent(arg)
    return intent in {"out", "inout"}


def build_value_map(proc: ft.Procedure) -> Dict[str, str]:
    """Create mapping from Fortran argument names to C variable names."""
    mapping: Dict[str, str] = {}
    for arg in proc.arguments:
        if is_hidden_argument(arg):
            mapping[arg.name] = f"{arg.name}_val"
            continue

        if is_array(arg):
            if should_parse_argument(arg):
                mapping[arg.name] = f"{arg.name}_arr"
            continue

        if is_derived_type(arg):
            mapping[arg.name] = derived_pointer_name(arg.name)
            continue

        if arg.type.lower().startswith("character"):
            mapping[arg.name] = arg.name
        else:
            mapping[arg.name] = f"{arg.name}_val"
    return mapping


def dimension_c_expression(expr: str, value_map: Dict[str, str]) -> str:
    """Convert a Fortran dimension expression into C code."""
    expression = expr.strip()
    if not expression:
        return "0"

    # Handle explicit lower:upper ranges by converting to a length expression.
    if ":" in expression:
        stripped = expression.replace(" ", "")
        if stripped not in {":", "::"}:
            lower_raw, upper_raw = expression.split(":", 1)
            lower_raw = lower_raw.strip()
            upper_raw = upper_raw.strip()
            # Default the lower bound to 1 when omitted (e.g. ':n').
            lower_expr = lower_raw if lower_raw else "1"
            if not upper_raw:
                # Cannot determine the extent without an upper bound; fall back to zero so caller raises.
                return "0"
            lower_c = dimension_c_expression(lower_expr, value_map) if lower_expr != expression else lower_expr
            upper_c = dimension_c_expression(upper_raw, value_map) if upper_raw != expression else upper_raw
            return f"(({upper_c}) - ({lower_c}) + 1)"

    def replace_size(match):
        array_name = match.group(1)
        dim_index = match.group(2)
        arr_var = value_map.get(array_name)
        if not arr_var:
            return match.group(0)
        try:
            dim_val = int(dim_index) - 1
            if dim_val < 0:
                dim_val = 0
            return f"PyArray_DIM({arr_var}, {dim_val})"
        except ValueError:
            dim_expr = dimension_c_expression(dim_index, value_map)
            return f"PyArray_DIM({arr_var}, (({dim_expr}) - 1))"

    expression = re.sub(r"size\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([0-9]+)\s*\)", replace_size, expression)

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)
    value_map_lower = {name.lower(): value for name, value in value_map.items()}
    for token in tokens:
        replacement = value_map_lower.get(token.lower())
        if replacement:
            escaped = re.escape(token)
            pattern = r"\b%s\b" % escaped
            expression = re.sub(pattern, replacement, expression, flags=re.IGNORECASE)
    return expression


def original_dimensions(
    proc: Optional[ft.Procedure], name: str, shape_hints: Optional[Dict[Tuple[str, Optional[str], str, str], List[str]]]
) -> Optional[List[str]]:
    """Look up original dimension expressions before wrapper generation."""
    if proc is None or not shape_hints:
        return None
    key = (proc.mod_name, getattr(proc, "type_name", None), proc.name, name)
    dims = shape_hints.get(key)
    if dims is not None:
        return dims
    return None


def extract_dimensions(arg: ft.Argument) -> List[str]:
    """Extract array dimensions from argument attributes."""
    for attr in arg.attributes:
        if attr.startswith("dimension("):
            dim_str = attr[len("dimension("):-1]
            return [d.strip() for d in dim_str.split(",")]
    return []