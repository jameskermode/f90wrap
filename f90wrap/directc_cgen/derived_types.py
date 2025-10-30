"""Derived type handling for Direct-C code generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from f90wrap.numpy_utils import build_arg_format, c_type_from_fortran, parse_arg_format
from .utils import ModuleHelper, character_length_expr

if TYPE_CHECKING:
    from . import DirectCGenerator


def write_type_member_get_wrapper(gen: DirectCGenerator, helper: ModuleHelper, helper_symbol: str) -> None:
    """Write getter wrapper for derived type members."""
    fmt = build_arg_format(helper.element.type)
    gen.write("PyObject* py_handle;")
    gen.write("static char *kwlist[] = {\"handle\", NULL};")
    gen.write("if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"O\", kwlist, &py_handle)) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    _extract_type_handle(gen)

    if fmt == "s":
        _write_character_type_getter(gen, helper, helper_symbol)
        return

    c_type = c_type_from_fortran(helper.element.type, gen.kind_map)
    gen.write(f"{c_type} value;")
    gen.write(f"{helper_symbol}(this_handle, &value);")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    if fmt == "O":
        gen.write("return PyBool_FromLong(value);")
    else:
        gen.write(f"return Py_BuildValue(\"{fmt}\", value);")


def _extract_type_handle(gen: DirectCGenerator) -> None:
    """Helper to extract handle from Python object for type members."""
    gen.write("PyObject* handle_sequence = PySequence_Fast(py_handle, \"Handle must be a sequence\");")
    gen.write("if (handle_sequence == NULL) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);")
    gen.write(f"if (handle_len != {gen.handle_size}) {{")
    gen.indent()
    gen.write("Py_DECREF(handle_sequence);")
    gen.write("PyErr_SetString(PyExc_ValueError, \"Unexpected handle length\");")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"int this_handle[{gen.handle_size}] = {{0}};")
    gen.write(f"for (int i = 0; i < {gen.handle_size}; ++i) {{")
    gen.indent()
    gen.write("PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);")
    gen.write("if (item == NULL) {")
    gen.indent()
    gen.write("Py_DECREF(handle_sequence);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("this_handle[i] = (int)PyLong_AsLong(item);")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write("Py_DECREF(handle_sequence);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("}")
    gen.write("Py_DECREF(handle_sequence);")


def _write_character_type_getter(gen: DirectCGenerator, helper: ModuleHelper, helper_symbol: str) -> None:
    """Helper to write character type getter for type members."""
    length_expr = character_length_expr(helper.element.type) or "1024"
    gen.write(f"int value_len = {length_expr};")
    gen.write("if (value_len <= 0) {")
    gen.indent()
    gen.write(
        "PyErr_SetString(PyExc_ValueError, \"Character helper length must be positive\");"
    )
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("char* buffer = (char*)malloc((size_t)value_len + 1);")
    gen.write("if (buffer == NULL) {")
    gen.indent()
    gen.write("PyErr_NoMemory();")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("memset(buffer, ' ', value_len);")
    gen.write("buffer[value_len] = '\\0';")
    gen.write(f"{helper_symbol}(this_handle, buffer, value_len);")
    gen.write("int actual_len = value_len;")
    gen.write("while (actual_len > 0 && buffer[actual_len - 1] == ' ') {")
    gen.indent()
    gen.write("--actual_len;")
    gen.dedent()
    gen.write("}")
    gen.write("PyObject* result = PyBytes_FromStringAndSize(buffer, actual_len);")
    gen.write("free(buffer);")
    gen.write("return result;")


def write_type_member_set_wrapper(gen: DirectCGenerator, helper: ModuleHelper, helper_symbol: str) -> None:
    """Write setter wrapper for derived type members."""
    fmt = parse_arg_format(helper.element.type)
    if fmt == "s":
        _write_character_type_setter(gen, helper, helper_symbol)
        return

    c_type = c_type_from_fortran(helper.element.type, gen.kind_map)
    # Use double for Python parse (format "d"), then cast to actual C type if needed
    parse_type = "double" if fmt == "d" else c_type
    gen.write("PyObject* py_handle;")
    gen.write(f"{parse_type} value;")
    gen.write(f"static char *kwlist[] = {{\"handle\", \"{helper.element.name}\", NULL}};")
    gen.write(
        f"if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"O{fmt}\", kwlist, &py_handle, &value)) {{"
    )
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    _extract_type_handle(gen)

    # Cast if parse type differs from Fortran type
    if parse_type != c_type:
        gen.write(f"{c_type} fortran_value = ({c_type})value;")
        gen.write(f"{helper_symbol}(this_handle, &fortran_value);")
    else:
        gen.write(f"{helper_symbol}(this_handle, &value);")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("Py_RETURN_NONE;")


def _write_character_type_setter(gen: DirectCGenerator, helper: ModuleHelper, helper_symbol: str) -> None:
    """Helper to write character type setter for type members."""
    gen.write("PyObject* py_handle;")
    gen.write("PyObject* py_value;")
    gen.write(f"static char *kwlist[] = {{\"handle\", \"{helper.element.name}\", NULL}};")
    gen.write(
        "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"OO\", kwlist, &py_handle, &py_value)) {"
    )
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    _extract_type_handle(gen)

    gen.write("if (py_value == Py_None) {")
    gen.indent()
    gen.write(
        f'PyErr_SetString(PyExc_TypeError, "Argument {helper.element.name} must be str or bytes");'
    )
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("PyObject* value_bytes = NULL;")
    gen.write("if (PyBytes_Check(py_value)) {")
    gen.indent()
    gen.write("value_bytes = py_value;")
    gen.write("Py_INCREF(value_bytes);")
    gen.dedent()
    gen.write("} else if (PyUnicode_Check(py_value)) {")
    gen.indent()
    gen.write("value_bytes = PyUnicode_AsUTF8String(py_value);")
    gen.write("if (value_bytes == NULL) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("} else {")
    gen.indent()
    gen.write(
        f'PyErr_SetString(PyExc_TypeError, "Argument {helper.element.name} must be str or bytes");'
    )
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("int value_len = (int)PyBytes_GET_SIZE(value_bytes);")
    gen.write("char* value = (char*)malloc((size_t)value_len + 1);")
    gen.write("if (value == NULL) {")
    gen.indent()
    gen.write("Py_DECREF(value_bytes);")
    gen.write("PyErr_NoMemory();")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(
        "memcpy(value, PyBytes_AS_STRING(value_bytes), (size_t)value_len);"
    )
    gen.write("value[value_len] = '\\0';")
    gen.write(f"{helper_symbol}(this_handle, value, value_len);")
    gen.write("free(value);")
    gen.write("Py_DECREF(value_bytes);")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("Py_RETURN_NONE;")