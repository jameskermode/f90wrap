"""Argument parsing and preparation for Direct-C code generation."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from f90wrap import fortran as ft
from f90wrap.numpy_utils import c_type_from_fortran, numpy_type_from_fortran, parse_arg_format
from .arguments_array import (
    declare_array_storage,
    prepare_output_array,
    write_array_preparation,
)
from .arguments_scalar import (
    prepare_character_argument,
    prepare_scalar_argument,
)
from .utils import (
    is_hidden_argument,
    is_array,
    is_derived_type,
    should_parse_argument,
    arg_intent,
    is_optional,
    is_output_argument,
    derived_pointer_name,
    character_length_expr,
    extract_dimensions,
    dimension_c_expression,
    original_dimensions,
)

if TYPE_CHECKING:
    from . import DirectCGenerator


def write_arg_parsing(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Generate PyArg_ParseTuple code for procedure arguments."""
    hidden_args = [arg for arg in proc.arguments if is_hidden_argument(arg)]
    for hidden in hidden_args:
        gen.write(f"int {hidden.name}_val = 0;")

    if not proc.arguments:
        return

    format_parts: List[str] = []
    parse_vars: List[str] = []
    kw_names: List[str] = []
    optional_started = False

    for arg in proc.arguments:
        if is_hidden_argument(arg):
            # Hidden arguments (like f90wrap_n0 dimension vars) need to be in kwlist
            # so Python can pass them, but they're optional and use integer format
            if not optional_started:
                format_parts.append("|")
                optional_started = True
            format_parts.append("i")
            parse_vars.append(f"&{arg.name}_val")
            kw_names.append(f'"{arg.name}"')
            continue

        intent = arg_intent(arg)
        optional = is_optional(arg)
        should_parse = should_parse_argument(arg)

        if not should_parse:
            if not is_array(arg) and not is_derived_type(arg) and not arg.type.lower().startswith("character"):
                c_type = c_type_from_fortran(arg.type, gen.kind_map)
                gen.write(f"{c_type} {arg.name}_val = 0;")
            continue

        if optional and not optional_started:
            format_parts.append("|")
            optional_started = True

        if is_derived_type(arg):
            format_parts.append("O")
            gen.write(f"PyObject* py_{arg.name} = NULL;")
            parse_vars.append(f"&py_{arg.name}")
        elif is_array(arg):
            format_parts.append("O")
            gen.write(f"PyObject* py_{arg.name} = NULL;")
            parse_vars.append(f"&py_{arg.name}")
        elif arg.type.lower().startswith("character"):
            if should_parse_argument(arg):
                format_parts.append("O")
                if optional or intent != "in":
                    gen.write(f"PyObject* py_{arg.name} = Py_None;")
                else:
                    gen.write(f"PyObject* py_{arg.name} = NULL;")
                parse_vars.append(f"&py_{arg.name}")
            else:
                format_parts.append("O")
                if optional:
                    gen.write(f"PyObject* py_{arg.name} = Py_None;")
                else:
                    gen.write(f"PyObject* py_{arg.name} = NULL;")
                parse_vars.append(f"&py_{arg.name}")
        else:
            c_type = c_type_from_fortran(arg.type, gen.kind_map)
            format_parts.append("O")
            if optional:
                gen.write(f"PyObject* py_{arg.name} = Py_None;")
            else:
                gen.write(f"PyObject* py_{arg.name} = NULL;")
            parse_vars.append(f"&py_{arg.name}")
            gen.write(f"{c_type} {arg.name}_val = 0;")
            gen.write(f"PyArrayObject* {arg.name}_scalar_arr = NULL;")
            gen.write(f"int {arg.name}_scalar_copyback = 0;")
            gen.write(f"int {arg.name}_scalar_is_array = 0;")

        kw_names.append(f'"{arg.name}"')

    if parse_vars:
        format_str = "".join(format_parts) if format_parts else ""
        kwlist = ", ".join(kw_names) if kw_names else ""
        gen.write(f"static char *kwlist[] = {{{kwlist}{', ' if kwlist else ''}NULL}};")
        gen.write("")
        gen.write(
            f'if (!PyArg_ParseTupleAndKeywords(args, kwargs, "{format_str}", kwlist, '
            f"{', '.join(parse_vars)})) {{"
        )
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.write("")


def write_arg_preparation(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Prepare arguments for helper function call."""
    for arg in proc.arguments:
        intent = arg_intent(arg)
        optional = is_optional(arg)
        parsed = should_parse_argument(arg)

        if is_array(arg):
            declare_array_storage(gen, arg)
            if parsed:
                if optional:
                    gen.write(f"if (py_{arg.name} != NULL && py_{arg.name} != Py_None) {{")
                    gen.indent()
                write_array_preparation(gen, arg)
                if optional:
                    gen.dedent()
                    gen.write("}")
            else:
                prepare_output_array(gen, arg)
        elif arg.type.lower().startswith("character"):
            prepare_character_argument(gen, arg, intent, optional)
        elif is_derived_type(arg):
            if should_parse_argument(arg):
                if optional:
                    ptr_name = derived_pointer_name(arg.name)
                    gen.write(f"PyObject* {arg.name}_handle_obj = NULL;")
                    gen.write(f"PyObject* {arg.name}_sequence = NULL;")
                    gen.write(f"Py_ssize_t {arg.name}_handle_len = 0;")
                    gen.write(f"int* {ptr_name} = NULL;")

                    gen.write(f"if (py_{arg.name} != Py_None) {{")
                    gen.indent()
                    gen.write(f"if (PyObject_HasAttrString(py_{arg.name}, \"_handle\")) {{")
                    gen.indent()
                    gen.write(f"{arg.name}_handle_obj = PyObject_GetAttrString(py_{arg.name}, \"_handle\");")
                    gen.write(f"if ({arg.name}_handle_obj == NULL) {{")
                    gen.indent()
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")
                    gen.write(
                        f"{arg.name}_sequence = PySequence_Fast({arg.name}_handle_obj, \"Failed to access handle sequence\");"
                    )
                    gen.write(f"if ({arg.name}_sequence == NULL) {{")
                    gen.indent()
                    gen.write(f"Py_DECREF({arg.name}_handle_obj);")
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")
                    gen.dedent()
                    gen.write(f"}} else if (PySequence_Check(py_{arg.name})) {{")
                    gen.indent()
                    gen.write(
                        f"{arg.name}_sequence = PySequence_Fast(py_{arg.name}, \"Argument {arg.name} must be a handle sequence\");"
                    )
                    gen.write(f"if ({arg.name}_sequence == NULL) {{")
                    gen.indent()
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")
                    gen.dedent()
                    gen.write("} else {")
                    gen.indent()
                    gen.write(
                        f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} must be a Fortran derived-type instance");'
                    )
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")

                    gen.write(f"{arg.name}_handle_len = PySequence_Fast_GET_SIZE({arg.name}_sequence);")
                    gen.write(f"if ({arg.name}_handle_len != {gen.handle_size}) {{")
                    gen.indent()
                    gen.write(
                        f'PyErr_SetString(PyExc_ValueError, "Argument {arg.name} has an invalid handle length");'
                    )
                    gen.write(f"Py_DECREF({arg.name}_sequence);")
                    gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")

                    gen.write(f"{ptr_name} = (int*)malloc(sizeof(int) * {arg.name}_handle_len);")
                    gen.write(f"if ({ptr_name} == NULL) {{")
                    gen.indent()
                    gen.write("PyErr_NoMemory();")
                    gen.write(f"Py_DECREF({arg.name}_sequence);")
                    gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")

                    gen.write(f"for (Py_ssize_t i = 0; i < {arg.name}_handle_len; ++i) {{")
                    gen.indent()
                    gen.write(f"PyObject* item = PySequence_Fast_GET_ITEM({arg.name}_sequence, i);")
                    gen.write("if (item == NULL) {")
                    gen.indent()
                    gen.write(f"free({ptr_name});")
                    gen.write(f"Py_DECREF({arg.name}_sequence);")
                    gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")
                    gen.write(f"{ptr_name}[i] = (int)PyLong_AsLong(item);")
                    gen.write("if (PyErr_Occurred()) {")
                    gen.indent()
                    gen.write(f"free({ptr_name});")
                    gen.write(f"Py_DECREF({arg.name}_sequence);")
                    gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")
                    gen.dedent()
                    gen.write("}")
                    gen.write(f"(void){arg.name}_handle_len;  /* suppress unused warnings when unchanged */")

                    gen.dedent()
                    gen.write("}")
                else:
                    write_derived_preparation(gen, arg)
            else:
                gen.write(f"int {arg.name}[{gen.handle_size}] = {{0}};")
        else:
            prepare_scalar_argument(gen, arg, intent, optional)


def write_derived_preparation(gen: DirectCGenerator, arg: ft.Argument) -> None:
    """Extract derived-type handle from Python object."""
    name = arg.name
    ptr_name = derived_pointer_name(name)
    gen.write(f"PyObject* {name}_handle_obj = NULL;")
    gen.write(f"PyObject* {name}_sequence = NULL;")
    gen.write(f"Py_ssize_t {name}_handle_len = 0;")

    gen.write(f"if (PyObject_HasAttrString(py_{name}, \"_handle\")) {{")
    gen.indent()
    gen.write(f"{name}_handle_obj = PyObject_GetAttrString(py_{name}, \"_handle\");")
    gen.write(f"if ({name}_handle_obj == NULL) {{")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(
        f"{name}_sequence = PySequence_Fast({name}_handle_obj, \"Failed to access handle sequence\");"
    )
    gen.write(f"if ({name}_sequence == NULL) {{")
    gen.indent()
    gen.write(f"Py_DECREF({name}_handle_obj);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write(f"}} else if (PySequence_Check(py_{name})) {{")
    gen.indent()
    gen.write(
        f"{name}_sequence = PySequence_Fast(py_{name}, \"Argument {name} must be a handle sequence\");"
    )
    gen.write(f"if ({name}_sequence == NULL) {{")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("} else {")
    gen.indent()
    gen.write(
        f'PyErr_SetString(PyExc_TypeError, "Argument {name} must be a Fortran derived-type instance");'
    )
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write(
        f"{name}_handle_len = PySequence_Fast_GET_SIZE({name}_sequence);")
    gen.write(f"if ({name}_handle_len != {gen.handle_size}) {{")
    gen.indent()
    gen.write(
        f'PyErr_SetString(PyExc_ValueError, "Argument {name} has an invalid handle length");'
    )
    gen.write(f"Py_DECREF({name}_sequence);")
    gen.write(f"if ({name}_handle_obj) Py_DECREF({name}_handle_obj);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write(f"int* {ptr_name} = (int*)malloc(sizeof(int) * {name}_handle_len);")
    gen.write(f"if ({ptr_name} == NULL) {{")
    gen.indent()
    gen.write("PyErr_NoMemory();")
    gen.write(f"Py_DECREF({name}_sequence);")
    gen.write(f"if ({name}_handle_obj) Py_DECREF({name}_handle_obj);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write(f"for (Py_ssize_t i = 0; i < {name}_handle_len; ++i) {{")
    gen.indent()
    gen.write(
        f"PyObject* item = PySequence_Fast_GET_ITEM({name}_sequence, i);")
    gen.write("if (item == NULL) {")
    gen.indent()
    gen.write(f"free({ptr_name});")
    gen.write(f"Py_DECREF({name}_sequence);")
    gen.write(f"if ({name}_handle_obj) Py_DECREF({name}_handle_obj);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"{ptr_name}[i] = (int)PyLong_AsLong(item);")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write(f"free({ptr_name});")
    gen.write(f"Py_DECREF({name}_sequence);")
    gen.write(f"if ({name}_handle_obj) Py_DECREF({name}_handle_obj);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("}")
    gen.write(f"(void){name}_handle_len;  /* suppress unused warnings when unchanged */")
    gen.write("")
