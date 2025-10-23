"""Scalar and character argument helpers for Direct-C code generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from f90wrap import fortran as ft
from f90wrap.numpy_utils import c_type_from_fortran, numpy_type_from_fortran, parse_arg_format
from .utils import (
    character_length_expr,
    is_output_argument,
    should_parse_argument,
)

if TYPE_CHECKING:
    from . import DirectCGenerator


def _write_scalar_array_handling(gen: 'DirectCGenerator', arg: ft.Argument, c_type: str, numpy_type: str) -> None:
    """Handle NumPy array inputs for scalar arguments."""
    gen.write(f"if (PyArray_Check(py_{arg.name})) {{")
    gen.indent()
    gen.write(
        f"{arg.name}_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(\n"
        f"    py_{arg.name}, {numpy_type}, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);")
    gen.write(f"if ({arg.name}_scalar_arr == NULL) {{")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"if (PyArray_SIZE({arg.name}_scalar_arr) != 1) {{")
    gen.indent()
    gen.write(
        f'PyErr_SetString(PyExc_ValueError, "Argument {arg.name} must have exactly one element");'
    )
    gen.write(f"Py_DECREF({arg.name}_scalar_arr);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"{arg.name}_scalar_is_array = 1;")
    gen.write(f"{arg.name} = ({c_type}*)PyArray_DATA({arg.name}_scalar_arr);")
    gen.write(f"{arg.name}_val = {arg.name}[0];")
    gen.write(
        f"if (PyArray_DATA({arg.name}_scalar_arr) != PyArray_DATA((PyArrayObject*)py_{arg.name}) || "
        f"PyArray_TYPE({arg.name}_scalar_arr) != PyArray_TYPE((PyArrayObject*)py_{arg.name})) {{"
    )
    gen.indent()
    gen.write(f"{arg.name}_scalar_copyback = 1;")
    gen.dedent()
    gen.write("}")
    gen.dedent()


def _write_scalar_number_handling(gen: 'DirectCGenerator', arg: ft.Argument, c_type: str) -> None:
    """Handle plain numeric inputs for scalar arguments."""
    gen.write(f"}} else if (PyNumber_Check(py_{arg.name})) {{")
    gen.indent()
    fmt = parse_arg_format(arg.type)
    if fmt in {"i", "l", "h", "I"}:
        gen.write(f"{arg.name}_val = ({c_type})PyLong_AsLong(py_{arg.name});")
    elif fmt in {"k", "K"}:
        gen.write(f"{arg.name}_val = ({c_type})PyLong_AsUnsignedLong(py_{arg.name});")
    elif fmt in {"L", "q"}:
        gen.write(f"{arg.name}_val = ({c_type})PyLong_AsLongLong(py_{arg.name});")
    elif fmt == "Q":
        gen.write(f"{arg.name}_val = ({c_type})PyLong_AsUnsignedLongLong(py_{arg.name});")
    elif fmt in {"d", "f"}:
        gen.write(f"{arg.name}_val = ({c_type})PyFloat_AsDouble(py_{arg.name});")
    elif fmt == "p":
        gen.write(f"{arg.name}_val = ({c_type})PyObject_IsTrue(py_{arg.name});")
    else:
        gen.write(
            f'PyErr_SetString(PyExc_TypeError, "Unsupported argument {arg.name}");'
        )
        gen.write("return NULL;")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("} else {")
    gen.indent()
    gen.write(
        f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} must be a scalar number or NumPy array");'
    )
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")


def prepare_scalar_argument(gen: 'DirectCGenerator', arg: ft.Argument, intent: str, optional: bool) -> None:
    """Prepare scalar argument values."""
    c_type = c_type_from_fortran(arg.type, gen.kind_map)
    numpy_type = numpy_type_from_fortran(arg.type, gen.kind_map)

    if not should_parse_argument(arg):
        return

    gen.write(f"{c_type}* {arg.name} = &{arg.name}_val;")

    if optional:
        gen.write(f"if (py_{arg.name} == Py_None) {{")
        gen.indent()
        gen.write(f"{arg.name}_val = 0;")
        gen.dedent()
        gen.write("} else {")
        gen.indent()

    _write_scalar_array_handling(gen, arg, c_type, numpy_type)
    _write_scalar_number_handling(gen, arg, c_type)

    if optional:
        gen.dedent()
        gen.write("}")


def _prepare_character_none_case(
    gen: 'DirectCGenerator', arg: ft.Argument, intent: str, optional: bool, default_len: str
) -> None:
    """Handle None value for character arguments."""
    gen.write(f"if (py_{arg.name} == Py_None) {{")
    gen.indent()
    if optional or intent != "in":
        gen.write(f"{arg.name}_len = {default_len};")
        gen.write(f"if ({arg.name}_len <= 0) {{")
        gen.indent()
        gen.write(
            f'PyErr_SetString(PyExc_ValueError, "Character length for {arg.name} must be positive");'
        )
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.write(f"{arg.name} = (char*)malloc((size_t){arg.name}_len + 1);")
        gen.write(f"if ({arg.name} == NULL) {{")
        gen.indent()
        gen.write("PyErr_NoMemory();")
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.write(f"memset({arg.name}, ' ', {arg.name}_len);")
        gen.write(f"{arg.name}[{arg.name}_len] = '\\0';")
    else:
        gen.write(
            f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} cannot be None");'
        )
        gen.write("return NULL;")
    gen.dedent()


def _prepare_character_string_case(gen: 'DirectCGenerator', arg: ft.Argument, is_output: bool) -> None:
    """Handle string/bytes/array values for character arguments."""
    gen.write("} else {")
    gen.indent()
    gen.write(f"PyObject* {arg.name}_bytes = NULL;")
    gen.write(f"if (PyArray_Check(py_{arg.name})) {{")
    gen.indent()
    gen.write("/* Handle numpy array - extract buffer for in-place modification */")
    gen.write(f"PyArrayObject* {arg.name}_arr = (PyArrayObject*)py_{arg.name};")
    gen.write(f"if (PyArray_TYPE({arg.name}_arr) != NPY_STRING) {{")
    gen.indent()
    gen.write(
        f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} must be a string array");'
    )
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"{arg.name}_len = (int)PyArray_ITEMSIZE({arg.name}_arr);")
    gen.write(f"{arg.name} = (char*)PyArray_DATA({arg.name}_arr);")
    if is_output:
        gen.write(f"{arg.name}_is_array = 1;")
    gen.dedent()
    gen.write(f"}} else if (PyBytes_Check(py_{arg.name})) {{")
    gen.indent()
    gen.write(f"{arg.name}_bytes = py_{arg.name};")
    gen.write(f"Py_INCREF({arg.name}_bytes);")
    gen.dedent()
    gen.write(f"}} else if (PyUnicode_Check(py_{arg.name})) {{")
    gen.indent()
    gen.write(f"{arg.name}_bytes = PyUnicode_AsUTF8String(py_{arg.name});")
    gen.write(f"if ({arg.name}_bytes == NULL) {{")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("} else {")
    gen.indent()
    gen.write(
        f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} must be str, bytes, or numpy array");'
    )
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"if ({arg.name}_bytes != NULL) {{")
    gen.indent()
    gen.write(f"{arg.name}_len = (int)PyBytes_GET_SIZE({arg.name}_bytes);")
    gen.write(f"{arg.name} = (char*)malloc((size_t){arg.name}_len + 1);")
    gen.write(f"if ({arg.name} == NULL) {{")
    gen.indent()
    gen.write(f"Py_DECREF({arg.name}_bytes);")
    gen.write("PyErr_NoMemory();")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(
        f"memcpy({arg.name}, PyBytes_AS_STRING({arg.name}_bytes), (size_t){arg.name}_len);"
    )
    gen.write(f"{arg.name}[{arg.name}_len] = '\\0';")
    gen.write(f"Py_DECREF({arg.name}_bytes);")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("}")


def prepare_character_argument(gen: 'DirectCGenerator', arg: ft.Argument, intent: str, optional: bool) -> None:
    """Allocate buffers and parse character arguments."""
    default_len = character_length_expr(arg.type) or "1024"

    if should_parse_argument(arg):
        gen.write(f"int {arg.name}_len = 0;")
        gen.write(f"char* {arg.name} = NULL;")
        gen.write(f"int {arg.name}_is_array = 0;")
        _prepare_character_none_case(gen, arg, intent, optional, default_len)
        _prepare_character_string_case(gen, arg, True)
    else:
        gen.write(f"int {arg.name}_len = {default_len};")
        gen.write(f"if ({arg.name}_len <= 0) {{")
        gen.indent()
        gen.write(
            f'PyErr_SetString(PyExc_ValueError, "Character length for {arg.name} must be positive");'
        )
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.write(f"char* {arg.name} = (char*)malloc((size_t){arg.name}_len + 1);")
        gen.write(f"if ({arg.name} == NULL) {{")
        gen.indent()
        gen.write("PyErr_NoMemory();")
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.write(f"memset({arg.name}, ' ', {arg.name}_len);")
        gen.write(f"{arg.name}[{arg.name}_len] = '\\0';")
