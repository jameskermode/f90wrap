"""NumPy C API utilities for Direct-C code generation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple


def _normalize_fortran_type(ftype: str) -> Tuple[str, Optional[str], Dict[str, bool]]:
    """Normalize Fortran type spelling to (base, kind, modifiers)."""

    modifiers: Dict[str, bool] = {}

    ftype_lower = ftype.strip().lower()
    base, _, kind_str = ftype_lower.partition("(")
    base = base.strip()

    if kind_str:
        kind_str = kind_str.rstrip(")").strip() or None
    else:
        kind_str = None

    if base.startswith("character"):
        base = "character"

    if "*" in base:
        stem, star = base.split("*", 1)
        stem = stem.strip()
        star = star.strip()
        if stem in {"real", "integer", "complex", "logical", "character"}:
            base = stem
            if not kind_str and star:
                kind_str = star

    if base == "double precision":
        base = "real"
        modifiers["force_double"] = True
        if not kind_str:
            kind_str = "8"

    return base, kind_str, modifiers


def numpy_type_from_fortran(ftype: str, kind_map: Dict[str, Dict[str, str]]) -> str:
    """Convert Fortran type to NumPy dtype enum constant."""

    base, kind_str, modifiers = _normalize_fortran_type(ftype)
    force_double = modifiers.get("force_double", False)

    # Map basic types
    if base == "integer":
        if kind_str and base in kind_map and kind_str in kind_map[base]:
            c_type = kind_map[base][kind_str]
            if c_type == "int":
                return "NPY_INT32"
            elif c_type == "long_long":
                return "NPY_INT64"
        if kind_str and kind_str.isdigit():
            if int(kind_str) >= 8:
                return "NPY_INT64"
            if int(kind_str) <= 2:
                return "NPY_INT16"
        return "NPY_INT"

    elif base == "real":
        if kind_str and base in kind_map and kind_str in kind_map[base]:
            c_type = kind_map[base][kind_str]
            if c_type == "float":
                return "NPY_FLOAT32"
            elif c_type == "double":
                return "NPY_FLOAT64"
        if kind_str and kind_str.isdigit():
            bits = int(kind_str)
            if bits >= 8:
                return "NPY_FLOAT64"
            if bits <= 4:
                return "NPY_FLOAT32"
        if force_double:
            return "NPY_FLOAT64"
        return "NPY_FLOAT32"

    elif base == "logical":
        # Fortran logical is typically 4 bytes (same as integer), so it maps
        # to int in C (see c_type_from_fortran). Use NPY_INT32 to match.
        # NumPy bool is only 1 byte and would cause a size mismatch. See #307.
        return "NPY_INT32"

    elif base == "complex":
        complex_map = {
            "complex_float": "NPY_COMPLEX64",
            "float_complex": "NPY_COMPLEX64",
            "complex_double": "NPY_COMPLEX128",
            "double_complex": "NPY_COMPLEX128",
            "complex_long_double": "NPY_CLONGDOUBLE",
            "long_double_complex": "NPY_CLONGDOUBLE",
        }
        if kind_str and base in kind_map and kind_str in kind_map[base]:
            c_type = kind_map[base][kind_str].lower()
            if c_type in complex_map:
                return complex_map[c_type]
        if kind_str and kind_str.isdigit():
            bits = int(kind_str)
            if bits >= 16:
                return "NPY_CLONGDOUBLE"
            if bits >= 8:
                return "NPY_CDOUBLE"
        return "NPY_CDOUBLE"

    elif base == "character":
        return "NPY_STRING"

    return "NPY_OBJECT"  # fallback for unknown types


def c_type_from_fortran(ftype: str, kind_map: Dict[str, Dict[str, str]]) -> str:
    """Convert Fortran type to C type string."""

    base, kind_str, modifiers = _normalize_fortran_type(ftype)
    force_double = modifiers.get("force_double", False)

    # Map basic types
    if base == "integer":
        if kind_str and base in kind_map and kind_str in kind_map[base]:
            c_type = kind_map[base][kind_str]
            if c_type == "int":
                return "int"
            elif c_type == "long_long":
                return "long long"
        if kind_str and kind_str.isdigit():
            bits = int(kind_str)
            if bits >= 8:
                return "long long"
            if bits <= 2:
                return "short"
        return "int"

    elif base == "real":
        if kind_str and base in kind_map and kind_str in kind_map[base]:
            c_type = kind_map[base][kind_str]
            if c_type == "float":
                return "float"
            elif c_type == "double":
                return "double"
        if kind_str and kind_str.isdigit():
            bits = int(kind_str)
            if bits >= 8:
                return "double"
            if bits <= 4:
                return "float"
        if force_double:
            return "double"
        return "float"

    elif base == "logical":
        return "int"  # Fortran logical maps to int in C

    elif base == "complex":
        complex_map = {
            "complex_float": "float _Complex",
            "float_complex": "float _Complex",
            "complex_double": "double _Complex",
            "double_complex": "double _Complex",
            "complex_long_double": "long double _Complex",
            "long_double_complex": "long double _Complex",
        }
        if kind_str and base in kind_map and kind_str in kind_map[base]:
            c_type = kind_map[base][kind_str].lower()
            if c_type in complex_map:
                return complex_map[c_type]
        if kind_str and kind_str.isdigit():
            bits = int(kind_str)
            if bits >= 16:
                return "long double _Complex"
            if bits >= 8:
                return "double _Complex"
        return "double _Complex"

    elif base == "character":
        return "char"

    return "void"  # fallback


def parse_arg_format(arg_type: str) -> str:
    """Get Python argument format character for PyArg_ParseTuple."""

    base, _, _ = _normalize_fortran_type(arg_type)

    if base == "integer":
        return "i"
    elif base == "real":
        return "d"
    elif base == "logical":
        return "p"  # boolean
    elif base == "complex":
        return "D"  # complex number
    elif base == "character":
        return "s"
    else:
        return "O"  # generic object


def build_arg_format(arg_type: str) -> str:
    """Get Python build format character for Py_BuildValue."""

    base, _, _ = _normalize_fortran_type(arg_type)

    if base == "integer":
        return "i"
    elif base == "real":
        return "d"
    elif base == "logical":
        return "O"  # Use PyBool_FromLong
    elif base == "complex":
        return "D"
    elif base == "character":
        return "s"
    else:
        return "O"  # generic object
