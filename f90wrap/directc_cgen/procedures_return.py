"""Return-value helpers for Direct-C procedures."""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

from f90wrap import fortran as ft
from f90wrap.numpy_utils import build_arg_format, numpy_type_from_fortran
from .utils import (
    is_output_argument,
    is_array,
    is_derived_type,
    should_parse_argument,
    is_hidden_argument,
    is_optional,
    derived_pointer_name,
    extract_dimensions,
)

if TYPE_CHECKING:
    from . import DirectCGenerator

def _handle_scalar_copyback(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Handle scalar array copyback for input/output arguments."""
    for arg in proc.arguments:
        if not should_parse_argument(arg):
            continue
        if (
            not is_array(arg)
            and not is_derived_type(arg)
            and not arg.type.lower().startswith("character")
        ):
            gen.write(f"if ({arg.name}_scalar_is_array) {{")
            gen.indent()
            gen.write(f"if ({arg.name}_scalar_copyback) {{")
            gen.indent()
            gen.write(
                f"if (PyArray_CopyInto((PyArrayObject*)py_{arg.name}, {arg.name}_scalar_arr) < 0) {{"
            )
            gen.indent()
            gen.write(f"Py_DECREF({arg.name}_scalar_arr);")
            gen.write("return NULL;")
            gen.dedent()
            gen.write("}")
            gen.dedent()
            gen.write("}")
            gen.write(f"Py_DECREF({arg.name}_scalar_arr);")
            gen.dedent()
            gen.write("}")


def _handle_array_copyback(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Handle array copyback for input/output arguments."""
    for arg in proc.arguments:
        if not is_array(arg) or not should_parse_argument(arg):
            continue
        if is_output_argument(arg):
            gen.write(f"if ({arg.name}_needs_copyback) {{")
            gen.indent()
            gen.write(
                f"if (PyArray_CopyInto((PyArrayObject*)py_{arg.name}, {arg.name}_arr) < 0) {{"
            )
            gen.indent()
            gen.write(f"Py_DECREF({arg.name}_arr);")
            gen.write(f"Py_DECREF(py_{arg.name}_arr);")
            gen.write("return NULL;")
            gen.dedent()
            gen.write("}")
            gen.dedent()
            gen.write("}")
        # Use Py_XDECREF for optional arrays (may be NULL)
        if is_optional(arg):
            gen.write(f"Py_XDECREF({arg.name}_arr);")
        else:
            gen.write(f"Py_DECREF({arg.name}_arr);")


def _handle_function_return(gen: DirectCGenerator, proc: ft.Function) -> None:
    """Handle return value for function procedures."""
    ret_type = proc.ret_val.type.lower()
    if is_array(proc.ret_val):
        write_array_return(gen, proc.ret_val, "result")
    elif ret_type.startswith("logical"):
        gen.write("return PyBool_FromLong(result);")
    else:
        fmt = build_arg_format(proc.ret_val.type)
        gen.write(f'return Py_BuildValue("{fmt}", result);')

    # Clean up non-output buffers for functions
    value_map = getattr(gen, "_value_map", {})
    for arg in proc.arguments:
        if arg.type.lower().startswith("character") and not is_output_argument(arg):
            cleanup_var = value_map.get(arg.name, arg.name)
            gen.write(f"free({cleanup_var});")
        elif is_array(arg) and should_parse_argument(arg):
            gen.write(f"Py_DECREF({arg.name}_arr);")
        elif is_derived_type(arg):
            ptr_name = derived_pointer_name(arg.name)
            gen.write(f"if ({arg.name}_sequence) {{")
            gen.indent()
            gen.write(f"Py_DECREF({arg.name}_sequence);")
            gen.dedent()
            gen.write("}")
            gen.write(f"if ({arg.name}_handle_obj) {{")
            gen.indent()
            gen.write(f"Py_DECREF({arg.name}_handle_obj);")
            gen.dedent()
            gen.write("}")
            gen.write(f"free({ptr_name});")


def _prepare_output_objects(gen: DirectCGenerator, output_args: List[ft.Argument]) -> List[str]:
    """Prepare Python objects for output arguments."""
    result_objects: List[str] = []

    for arg in output_args:
        if is_array(arg):
            result_objects.append(f"py_{arg.name}_arr")
            continue

        if arg.type.lower().startswith("character"):
            _prepare_character_output(gen, arg)
            # For numpy arrays, py_<arg>_obj will be Py_None, which is fine in tuple
            result_objects.append(f"py_{arg.name}_obj")
        elif is_derived_type(arg):
            _prepare_derived_output(gen, arg)
            result_objects.append(f"py_{arg.name}_obj")
        else:
            if arg.type.lower().startswith("logical"):
                gen.write(
                    f"PyObject* py_{arg.name}_obj = PyBool_FromLong({arg.name}_val);"
                )
            else:
                fmt = build_arg_format(arg.type)
                gen.write(
                    f"PyObject* py_{arg.name}_obj = Py_BuildValue(\"{fmt}\", {arg.name}_val);"
                )
            gen.write(f"if (py_{arg.name}_obj == NULL) {{")
            gen.indent()
            gen.write("return NULL;")
            gen.dedent()
            gen.write("}")
            result_objects.append(f"py_{arg.name}_obj")

    return result_objects


def _prepare_character_output(gen: DirectCGenerator, arg: ft.Argument) -> None:
    """Prepare character output object."""
    parsed = should_parse_argument(arg)

    if parsed:
        # Check if buffer is from numpy array
        gen.write(f"PyObject* py_{arg.name}_obj = NULL;")
        gen.write(f"if ({arg.name}_is_array) {{")
        gen.indent()
        gen.write("/* Numpy array was modified in place, no return object or free needed */")
        gen.dedent()
        gen.write("} else {")
        gen.indent()

    gen.write(f"int {arg.name}_trim = {arg.name}_len;")
    gen.write(f"while ({arg.name}_trim > 0 && {arg.name}[{arg.name}_trim - 1] == ' ') {{")
    gen.indent()
    gen.write(f"--{arg.name}_trim;")
    gen.dedent()
    gen.write("}")

    if parsed:
        gen.write(
            f"py_{arg.name}_obj = PyBytes_FromStringAndSize({arg.name}, {arg.name}_trim);"
        )
        gen.write(f"free({arg.name});")
        gen.write(f"if (py_{arg.name}_obj == NULL) {{")
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.dedent()
        gen.write("}")
    else:
        gen.write(
            f"PyObject* py_{arg.name}_obj = PyBytes_FromStringAndSize({arg.name}, {arg.name}_trim);"
        )
        gen.write(f"free({arg.name});")
        gen.write(f"if (py_{arg.name}_obj == NULL) {{")
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")


def _prepare_derived_output(gen: DirectCGenerator, arg: ft.Argument) -> None:
    """Prepare derived type output object."""
    parsed = should_parse_argument(arg)
    ptr_name = derived_pointer_name(arg.name)
    gen.write(f"PyObject* py_{arg.name}_obj = PyList_New({gen.handle_size});")
    gen.write(f"if (py_{arg.name}_obj == NULL) {{")
    gen.indent()
    if parsed:
        gen.write(f"free({ptr_name});")
        gen.write(f"if ({arg.name}_sequence) Py_DECREF({arg.name}_sequence);")
        gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"for (int i = 0; i < {gen.handle_size}; ++i) {{")
    gen.indent()
    gen.write(f"PyObject* item = PyLong_FromLong((long){ptr_name}[i]);")
    gen.write("if (item == NULL) {")
    gen.indent()
    gen.write(f"Py_DECREF(py_{arg.name}_obj);")
    if parsed:
        gen.write(f"free({ptr_name});")
        gen.write(f"if ({arg.name}_sequence) Py_DECREF({arg.name}_sequence);")
        gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"PyList_SET_ITEM(py_{arg.name}_obj, i, item);")
    gen.dedent()
    gen.write("}")
    if parsed:
        gen.write(f"if (PyObject_HasAttrString(py_{arg.name}, \"_handle\")) {{")
        gen.indent()
        gen.write(f"Py_INCREF(py_{arg.name}_obj);")
        gen.write(f"if (PyObject_SetAttrString(py_{arg.name}, \"_handle\", py_{arg.name}_obj) < 0) {{")
        gen.indent()
        gen.write(f"Py_DECREF(py_{arg.name}_obj);")
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.dedent()
        gen.write("}")
        gen.write(f"if ({arg.name}_sequence) Py_DECREF({arg.name}_sequence);")
        gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
        gen.write(f"free({ptr_name});")


def _cleanup_non_output_buffers(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Clean up non-output buffers."""
    value_map = getattr(gen, "_value_map", {})
    for arg in proc.arguments:
        if arg.type.lower().startswith("character") and not is_array(arg) and not is_output_argument(arg):
            cleanup_var = value_map.get(arg.name, arg.name)
            # Only free if not from numpy array
            if should_parse_argument(arg):
                gen.write(f"if (!{arg.name}_is_array) free({cleanup_var});")
            else:
                gen.write(f"free({cleanup_var});")
        elif is_derived_type(arg) and not is_output_argument(arg):
            ptr_name = derived_pointer_name(arg.name)
            gen.write(f"if ({arg.name}_sequence) {{")
            gen.indent()
            gen.write(f"Py_DECREF({arg.name}_sequence);")
            gen.dedent()
            gen.write("}")
            gen.write(f"if ({arg.name}_handle_obj) {{")
            gen.indent()
            gen.write(f"Py_DECREF({arg.name}_handle_obj);")
            gen.dedent()
            gen.write("}")
            gen.write(f"free({ptr_name});")


def write_return_value(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Build and return the Python return value."""
    output_args = [
        arg for arg in proc.arguments if is_output_argument(arg)
    ]

    _handle_scalar_copyback(gen, proc)
    _handle_array_copyback(gen, proc)

    if isinstance(proc, ft.Function):
        _handle_function_return(gen, proc)
        return

    result_objects = _prepare_output_objects(gen, output_args)
    _cleanup_non_output_buffers(gen, proc)

    if not result_objects:
        gen.write("Py_RETURN_NONE;")
        return

    # Filter out NULL objects at runtime (numpy arrays modified in place)
    gen.write("/* Build result tuple, filtering out NULL objects */")
    gen.write("int result_count = 0;")
    for name in result_objects:
        gen.write(f"if ({name} != NULL) result_count++;")

    gen.write("if (result_count == 0) {")
    gen.indent()
    gen.write("Py_RETURN_NONE;")
    gen.dedent()
    gen.write("}")

    gen.write("if (result_count == 1) {")
    gen.indent()
    for name in result_objects:
        gen.write(f"if ({name} != NULL) return {name};")
    gen.dedent()
    gen.write("}")

    gen.write("PyObject* result_tuple = PyTuple_New(result_count);")
    gen.write("if (result_tuple == NULL) {")
    gen.indent()
    for name in result_objects:
        gen.write(f"if ({name} != NULL) Py_DECREF({name});")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("int tuple_index = 0;")
    for name in result_objects:
        gen.write(f"if ({name} != NULL) {{")
        gen.indent()
        gen.write(f"PyTuple_SET_ITEM(result_tuple, tuple_index++, {name});")
        gen.dedent()
        gen.write("}")
    gen.write("return result_tuple;")


def write_array_return(gen: DirectCGenerator, ret_val: ft.Argument, var_name: str) -> None:
    """Create NumPy array from returned Fortran array with error handling."""
    numpy_type = numpy_type_from_fortran(ret_val.type, gen.kind_map)
    dims = extract_dimensions(ret_val)
    ndim = len(dims) if dims else 1

    gen.write("/* Create NumPy array from result */")
    gen.write(f"npy_intp result_dims[{ndim}];")
    for i, dim in enumerate(dims or [1]):
        gen.write(f"result_dims[{i}] = {dim};")

    gen.write(f"PyObject* result_arr = PyArray_New(&PyArray_Type, {ndim}, result_dims,")
    gen.write(f"    {numpy_type}, NULL, (void*){var_name},")
    gen.write(f"    0, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_OWNDATA, NULL);")
    gen.write("if (result_arr == NULL) {")
    gen.indent()
    gen.write("/* Free owned data buffer on failure to avoid leaks */")
    gen.write(f"free({var_name});")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("return result_arr;")


def procedure_error_args(gen: DirectCGenerator, proc: ft.Procedure) -> Optional[Tuple[str, str]]:
    """Return error argument names when auto-raise is enabled."""
    if not gen.error_num_arg or not gen.error_msg_arg:
        return None

    names = {arg.name for arg in proc.arguments}
    if gen.error_num_arg in names and gen.error_msg_arg in names:
        return (gen.error_num_arg, gen.error_msg_arg)
    return None


def write_error_cleanup(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Free allocated resources before returning on error."""
    for arg in proc.arguments:
        if arg.type.lower().startswith("character") and not is_array(arg):
            # Only free if not from numpy array
            if should_parse_argument(arg):
                gen.write(f"if (!{arg.name}_is_array) free({arg.name});")
            else:
                gen.write(f"free({arg.name});")
        elif is_array(arg):
            if is_output_argument(arg):
                gen.write(f"Py_XDECREF(py_{arg.name}_arr);")
            else:
                gen.write(f"Py_XDECREF({arg.name}_arr);")
        elif is_derived_type(arg) and should_parse_argument(arg):
            ptr_name = derived_pointer_name(arg.name)
            gen.write(f"if ({arg.name}_sequence) Py_DECREF({arg.name}_sequence);")
            gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
            gen.write(f"free({ptr_name});")


def write_auto_raise_guard(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Emit error handling guard for auto-raise logic."""
    error_args = procedure_error_args(gen, proc)
    if not error_args:
        return

    num_name, msg_name = error_args
    num_var = f"{num_name}_val"
    msg_ptr = msg_name
    msg_len = f"{msg_name}_len"

    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    write_error_cleanup(gen, proc)
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write(f"if ({num_var} != 0) {{")
    gen.indent()
    gen.write(f"f90wrap_abort_({msg_ptr}, {msg_len});")
    write_error_cleanup(gen, proc)
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
