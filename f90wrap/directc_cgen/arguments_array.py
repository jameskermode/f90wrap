"""Array argument utilities for Direct-C wrapper generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from f90wrap import fortran as ft
from f90wrap.numpy_utils import c_type_from_fortran, numpy_type_from_fortran
from .utils import (
    character_length_expr,
    dimension_c_expression,
    extract_dimensions,
    is_output_argument,
    original_dimensions,
    should_parse_argument,
)

if TYPE_CHECKING:
    from . import DirectCGenerator


def declare_array_storage(gen: DirectCGenerator, arg: ft.Argument) -> None:
    """Declare local variables needed for array arguments."""
    c_type = c_type_from_fortran(arg.type, gen.kind_map)
    gen.write(f"PyArrayObject* {arg.name}_arr = NULL;")
    if is_output_argument(arg):
        gen.write(f"PyObject* py_{arg.name}_arr = NULL;")
        if should_parse_argument(arg):
            gen.write(f"int {arg.name}_needs_copyback = 0;")
    gen.write(f"{c_type}* {arg.name} = NULL;")
    if arg.type.lower().startswith("character"):
        gen.write(f"int {arg.name}_elem_len = 0;")


def write_array_preparation(gen: DirectCGenerator, arg: ft.Argument) -> None:
    """Extract array data from NumPy arrays."""
    numpy_type = numpy_type_from_fortran(arg.type, gen.kind_map)
    c_type = c_type_from_fortran(arg.type, gen.kind_map)

    gen.write(f"/* Extract {arg.name} array data */")
    gen.write(f"if (!PyArray_Check(py_{arg.name})) {{")
    gen.indent()
    gen.write(f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} must be a NumPy array");')
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write(f"{arg.name}_arr = (PyArrayObject*)PyArray_FROM_OTF(")
    gen.write(
        f"    py_{arg.name}, {numpy_type}, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);"
    )
    gen.write(f"if ({arg.name}_arr == NULL) {{")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write(f"{arg.name} = ({c_type}*)PyArray_DATA({arg.name}_arr);")

    if arg.type.lower().startswith("character"):
        gen.write(f"{arg.name}_elem_len = (int)PyArray_ITEMSIZE({arg.name}_arr);")

    dims = extract_dimensions(arg)
    if dims:
        for i in range(len(dims)):
            gen.write(f"int n{i}_{arg.name} = (int)PyArray_DIM({arg.name}_arr, {i});")
        for i, dim in enumerate(dims):
            dim_name = dim.strip()
            if dim_name and dim_name.startswith("f90wrap_"):
                gen.write(f"{dim_name}_val = n{i}_{arg.name};")

    if is_output_argument(arg):
        gen.write(f"Py_INCREF(py_{arg.name});")
        gen.write(f"py_{arg.name}_arr = py_{arg.name};")
        if should_parse_argument(arg):
            gen.write(
                f"if (PyArray_DATA({arg.name}_arr) != PyArray_DATA((PyArrayObject*)py_{arg.name}) || "
                f"PyArray_TYPE({arg.name}_arr) != PyArray_TYPE((PyArrayObject*)py_{arg.name})) {{"
            )
            gen.indent()
            gen.write(f"{arg.name}_needs_copyback = 1;")
            gen.dedent()
            gen.write("}")

    gen.write("")


def prepare_output_array(gen: DirectCGenerator, arg: ft.Argument) -> None:
    """Allocate NumPy array for output-only arguments."""
    proc = getattr(gen, "_current_proc", None)
    value_map = getattr(gen, "_value_map", {})
    trans_dims = extract_dimensions(arg)
    orig_dims = original_dimensions(proc, arg.name, gen.shape_hints)
    if not trans_dims and not orig_dims:
        trans_dims = ["1"]

    count = max(len(trans_dims), len(orig_dims or [])) or 1

    dim_vars = []
    for index in range(count):
        trans_token = trans_dims[index] if index < len(trans_dims) else None
        source_expr = orig_dims[index] if orig_dims and index < len(orig_dims) else None
        if not source_expr:
            source_expr = trans_token or "1"

        expr = dimension_c_expression(source_expr, value_map)
        size_var = f"{arg.name}_dim_{index}"
        gen.write(f"npy_intp {size_var} = (npy_intp)({expr});")
        gen.write(f"if ({size_var} <= 0) {{")
        gen.indent()
        gen.write(
            f'PyErr_SetString(PyExc_ValueError, "Dimension for {arg.name} must be positive");'
        )
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")

        if trans_token:
            token = trans_token.strip()
            hidden_lower = getattr(gen, "_hidden_names_lower", set())
            token_lower = token.lower()
            if token_lower in hidden_lower:
                replacement = value_map.get(token) or value_map.get(token_lower)
                if replacement:
                    gen.write(f"{replacement} = (int){size_var};")
        dim_vars.append(size_var)

    dims_array = f"{arg.name}_dims"
    gen.write(f"npy_intp {dims_array}[{len(dim_vars)}] = {{{', '.join(dim_vars)}}};")
    numpy_type = numpy_type_from_fortran(arg.type, gen.kind_map)
    char_len = character_length_expr(arg.type) if arg.type.lower().startswith("character") else None
    if char_len is not None:
        # For character arrays with a known length, create a numpy array with the
        # correct item size so the Fortran function receives a buffer large enough
        # to hold all characters (dim * char_len bytes).
        # Use PyArray_DescrConverter with "S{N}" to avoid direct elsize mutation,
        # which is not supported in NumPy 2.x.
        gen.write(f"PyObject* {arg.name}_dtype_str = PyUnicode_FromString(\"S{char_len}\");")
        gen.write(f"if ({arg.name}_dtype_str == NULL) return NULL;")
        gen.write(f"PyArray_Descr* {arg.name}_descr = NULL;")
        gen.write(f"if (!PyArray_DescrConverter({arg.name}_dtype_str, &{arg.name}_descr)) {{")
        gen.indent()
        gen.write(f"Py_DECREF({arg.name}_dtype_str);")
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.write(f"Py_DECREF({arg.name}_dtype_str);")
        gen.write(
            f"py_{arg.name}_arr = PyArray_NewFromDescr(&PyArray_Type, {arg.name}_descr,"
            f" {len(dim_vars)}, {dims_array}, NULL, NULL, 1, NULL);"
        )
    else:
        # Use PyArray_EMPTY with fortran=1 for Fortran-contiguous (column-major) layout
        gen.write(
            f"py_{arg.name}_arr = PyArray_EMPTY({len(dim_vars)}, {dims_array}, {numpy_type}, 1);"
        )
    gen.write(f"if (py_{arg.name}_arr == NULL) {{")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"{arg.name}_arr = (PyArrayObject*)py_{arg.name}_arr;")
    c_type = c_type_from_fortran(arg.type, gen.kind_map)
    gen.write(f"{arg.name} = ({c_type}*)PyArray_DATA({arg.name}_arr);")
    if char_len is not None:
        gen.write(f"{arg.name}_elem_len = {char_len};")
    gen.write("")

