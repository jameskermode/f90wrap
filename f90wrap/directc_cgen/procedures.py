"""Procedure wrappers and return value handling for Direct-C code generation."""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

from f90wrap import fortran as ft
from f90wrap.numpy_utils import build_arg_format, c_type_from_fortran, numpy_type_from_fortran
from .procedures_return import (
    procedure_error_args,
    write_auto_raise_guard,
    write_error_cleanup,
    write_return_value,
)
from .utils import (
    is_output_argument,
    is_array,
    is_derived_type,
    should_parse_argument,
    is_hidden_argument,
    is_optional,
    derived_pointer_name,
    build_value_map,
    helper_symbol,
    helper_param_list,
    wrapper_name,
    extract_dimensions,
)
from .arguments import write_arg_parsing, write_arg_preparation

if TYPE_CHECKING:
    from . import DirectCGenerator


def write_wrapper_function(gen: DirectCGenerator, proc: ft.Procedure, mod_name: str) -> None:
    """Write Python C API wrapper function for a procedure."""
    proc_attributes = getattr(proc, "attributes", []) or []
    if 'destructor' in proc_attributes:
        write_destructor_wrapper(gen, proc, mod_name)
        return

    func_wrapper_name = wrapper_name(mod_name, proc)

    prev_value_map = getattr(gen, "_value_map", None)
    prev_hidden = getattr(gen, "_hidden_names", set())
    prev_hidden_lower = getattr(gen, "_hidden_names_lower", set())
    prev_proc = getattr(gen, "_current_proc", None)
    gen._value_map = build_value_map(proc)
    gen._hidden_names = {arg.name for arg in proc.arguments if is_hidden_argument(arg)}
    gen._hidden_names_lower = {name.lower() for name in gen._hidden_names}
    gen._current_proc = proc

    gen.write(
        f"static PyObject* {func_wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
    )
    gen.write("{")
    gen.indent()

    # Parse Python arguments
    write_arg_parsing(gen, proc)

    # Prepare arguments for helper call
    write_arg_preparation(gen, proc)

    # Call the helper function
    write_helper_call(gen, proc)

    if procedure_error_args(gen, proc):
        write_auto_raise_guard(gen, proc)

    # Build return value
    write_return_value(gen, proc)

    gen.dedent()
    gen.write("}")
    gen.write("")

    gen._value_map = prev_value_map
    gen._hidden_names = prev_hidden
    gen._hidden_names_lower = prev_hidden_lower
    gen._current_proc = prev_proc


def write_destructor_wrapper(gen: DirectCGenerator, proc: ft.Procedure, mod_name: str) -> None:
    """Specialised wrapper for derived-type destructors."""
    func_wrapper_name = wrapper_name(mod_name, proc)
    helper_sym = helper_symbol(proc, gen.prefix)
    arg = proc.arguments[0]

    gen.write(
        f"static PyObject* {func_wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
    )
    gen.write("{")
    gen.indent()
    gen.write("(void)self;")

    write_arg_parsing(gen, proc)
    write_arg_preparation(gen, proc)

    gen.write(f"/* Call f90wrap helper */")
    ptr_name = derived_pointer_name(arg.name)
    gen.write(f"{helper_sym}({ptr_name});")

    # Cleanup for derived handle
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

    gen.write("Py_RETURN_NONE;")
    gen.dedent()
    gen.write("}")
    gen.write("")


def write_helper_call(gen: DirectCGenerator, proc: ft.Procedure, helper_sym: Optional[str] = None) -> None:
    """Generate the call to the f90wrap helper function."""
    call_args = []
    char_lens = []  # Hidden length arguments come AFTER all explicit arguments
    helper_sym = helper_sym or helper_symbol(proc, gen.prefix)

    # Add result parameter for functions
    if isinstance(proc, ft.Function):
        c_type = c_type_from_fortran(proc.ret_val.type, gen.kind_map)
        if is_array(proc.ret_val):
            gen.write(f"{c_type}* result;")
        else:
            gen.write(f"{c_type} result;")
        call_args.append("&result")

    # Add regular arguments (explicit arguments first)
    for arg in proc.arguments:
        parsed = should_parse_argument(arg)
        if is_hidden_argument(arg):
            call_args.append(f"&{arg.name}_val")
        elif is_derived_type(arg):
            ptr_name = derived_pointer_name(arg.name)
            call_args.append(ptr_name)
        elif is_array(arg):
            call_args.append(arg.name)
            # For character arrays, we need to pass element length
            # (already computed in write_array_preparation)
            if arg.type.lower().startswith("character"):
                len_var = f"{arg.name}_elem_len"
                char_lens.append(len_var)
        elif arg.type.lower().startswith("character"):
            call_args.append(arg.name)
            # Save length for later - Fortran puts hidden lengths AFTER all explicit args
            char_lens.append(f"{arg.name}_len")
        else:
            if parsed:
                call_args.append(arg.name)
            else:
                call_args.append(f"&{arg.name}_val")

    # Add hidden character lengths at the end (Fortran calling convention)
    call_args.extend(char_lens)

    gen.write(f"/* Call f90wrap helper */")
    if call_args:
        gen.write(f"{helper_sym}({', '.join(call_args)});")
    else:
        gen.write(f"{helper_sym}();")

    # Check if Fortran code raised an exception via f90wrap_abort
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    write_error_cleanup(gen, proc)
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("")


def write_helper_declaration(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Write extern declaration for f90wrap helper functions."""
    helper_sym = helper_symbol(proc, gen.prefix)
    params = helper_param_list(proc, gen.kind_map)

    if params:
        param_str = ", ".join(params)
        gen.write(f"extern void {helper_sym}({param_str});")
    else:
        gen.write(f"extern void {helper_sym}(void);")


def write_alias_helper_declaration(
    gen: DirectCGenerator,
    alias_name: str,
    binding: ft.Binding,
    proc: ft.Procedure,
) -> None:
    """Write extern declaration for alias binding helpers."""
    from .utils import helper_param_list

    helper_sym = f"F90WRAP_F_SYMBOL({alias_name})"
    params = helper_param_list(proc, gen.kind_map)

    if params:
        param_str = ", ".join(params)
        gen.write(f"extern void {helper_sym}({param_str});")
    else:
        gen.write(f"extern void {helper_sym}(void);")
