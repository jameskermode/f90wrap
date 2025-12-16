"""Module-level helper wrappers for Direct-C code generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from f90wrap.numpy_utils import build_arg_format, c_type_from_fortran, parse_arg_format
from .module_helpers_array import (
    extract_parent_handle,
    write_array_helper_body,
    write_module_array_getitem_wrapper,
    write_module_array_len_wrapper,
    write_module_array_setitem_wrapper,
)
from .utils import (
    ModuleHelper,
    module_helper_wrapper_name,
    module_helper_symbol,
    character_length_expr,
)

if TYPE_CHECKING:
    from . import DirectCGenerator


def write_module_helper_wrapper(gen: DirectCGenerator, helper: ModuleHelper) -> None:
    """Emit specialised wrappers for module get/set/array helpers."""
    if helper.kind == "array_getitem":
        write_module_array_getitem_wrapper(gen, helper)
        return
    if helper.kind == "array_setitem":
        write_module_array_setitem_wrapper(gen, helper)
        return
    if helper.kind == "array_len":
        write_module_array_len_wrapper(gen, helper)
        return
    if helper.kind == "get_derived":
        write_module_get_derived_wrapper(gen, helper)
        return
    if helper.kind == "set_derived":
        write_module_set_derived_wrapper(gen, helper)
        return

    wrapper_name = module_helper_wrapper_name(helper)
    helper_symbol = module_helper_symbol(helper, gen.prefix)

    gen.write(
        f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
    )
    gen.write("{")
    gen.indent()
    gen.write("(void)self;")

    if helper.kind == "get":
        if helper.is_type_member:
            write_type_member_get_wrapper(gen, helper, helper_symbol)
        else:
            write_module_scalar_get_wrapper(gen, helper, helper_symbol)
        gen.dedent()
        gen.write("}")
        gen.write("")
        return

    if helper.kind == "set":
        if helper.is_type_member:
            write_type_member_set_wrapper(gen, helper, helper_symbol)
        else:
            write_module_scalar_set_wrapper(gen, helper, helper_symbol)
        gen.dedent()
        gen.write("}")
        gen.write("")
        return

    else:  # array helper
        write_array_helper_body(gen, helper, helper_symbol)
        gen.dedent()
        gen.write("}")
        gen.write("")


def write_module_scalar_get_wrapper(gen: DirectCGenerator, helper: ModuleHelper, helper_symbol: str) -> None:
    """Write scalar getter for module-level variable."""
    gen.write(
        "if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {"
    )
    gen.indent()
    gen.write(
        "PyErr_SetString(PyExc_TypeError, \"This helper does not accept arguments\");"
    )
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("")

    fmt = build_arg_format(helper.element.type)
    if fmt == "s":
        _write_character_getter(gen, helper, helper_symbol)
        return

    c_type = c_type_from_fortran(helper.element.type, gen.kind_map)
    gen.write(f"{c_type} value;")
    gen.write(f"{helper_symbol}(&value);")
    if fmt == "O":
        gen.write("return PyBool_FromLong(value);")
    else:
        gen.write(f"return Py_BuildValue(\"{fmt}\", value);")


def _write_character_getter(gen: DirectCGenerator, helper: ModuleHelper, helper_symbol: str) -> None:
    """Helper for character type getters."""
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
    gen.write(f"{helper_symbol}(buffer, value_len);")
    gen.write("int actual_len = value_len;")
    gen.write("while (actual_len > 0 && buffer[actual_len - 1] == ' ') {")
    gen.indent()
    gen.write("--actual_len;")
    gen.dedent()
    gen.write("}")
    gen.write("PyObject* result = PyBytes_FromStringAndSize(buffer, actual_len);")
    gen.write("free(buffer);")
    gen.write("if (result == NULL) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("return result;")


def write_module_scalar_set_wrapper(gen: DirectCGenerator, helper: ModuleHelper, helper_symbol: str) -> None:
    """Write scalar setter for module-level variable."""
    fmt = parse_arg_format(helper.element.type)
    if fmt == "s":
        _write_character_setter(gen, helper, helper_symbol)
        return

    if fmt == "D":
        gen.write("Py_complex value;")
        gen.write(f"static char *kwlist[] = {{\"{helper.element.name}\", NULL}};")
        gen.write(
            f"if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"{fmt}\", kwlist, &value)) {{"
        )
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        c_type = c_type_from_fortran(helper.element.type, gen.kind_map)
        gen.write(f"{c_type} fortran_value = ({c_type})(value.real + value.imag * I);")
        gen.write(f"{helper_symbol}(&fortran_value);")
        gen.write("if (PyErr_Occurred()) {")
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.write("Py_RETURN_NONE;")
        return

    c_type = c_type_from_fortran(helper.element.type, gen.kind_map)
    parse_type = "double" if fmt == "d" else c_type
    gen.write(f"{parse_type} value;")
    gen.write(f"static char *kwlist[] = {{\"{helper.element.name}\", NULL}};")
    gen.write(
        f"if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"{fmt}\", kwlist, &value)) {{"
    )
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    if parse_type != c_type:
        gen.write(f"{c_type} fortran_value = ({c_type})value;")
        gen.write(f"{helper_symbol}(&fortran_value);")
    else:
        gen.write(f"{helper_symbol}(&value);")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("Py_RETURN_NONE;")


def _write_character_setter(gen: DirectCGenerator, helper: ModuleHelper, helper_symbol: str) -> None:
    """Helper for character type setters."""
    gen.write("PyObject* py_value;")
    gen.write(f"static char *kwlist[] = {{\"{helper.element.name}\", NULL}};")
    gen.write(
        "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"O\", kwlist, &py_value)) {"
    )
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
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
    gen.write("memcpy(value, PyBytes_AS_STRING(value_bytes), (size_t)value_len);")
    gen.write("value[value_len] = '\\0';")
    gen.write(f"{helper_symbol}(value, value_len);")
    gen.write("free(value);")
    gen.write("Py_DECREF(value_bytes);")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("Py_RETURN_NONE;")
def _write_derived_getter_body(gen: DirectCGenerator, helper: ModuleHelper, helper_symbol: str) -> None:
    """Helper for derived type getter body."""
    if helper.is_type_member:
        gen.write("PyObject* py_handle;")
        gen.write("static char *kwlist[] = {\"handle\", NULL};")
        gen.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"O\", kwlist, &py_handle)) {"
        )
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        extract_parent_handle(gen, "handle")
        gen.write("Py_DECREF(handle_sequence);")
    else:
        gen.write("if (args && PyTuple_Size(args) != 0) {")
        gen.indent()
        gen.write("PyErr_SetString(PyExc_TypeError, \"Getters do not take arguments\");")
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")

    gen.write(f"int value_handle[{gen.handle_size}] = {{0}};")
    if helper.is_type_member:
        gen.write(f"{helper_symbol}(handle_handle, value_handle);")
    else:
        gen.write(f"{helper_symbol}(value_handle);")


def write_module_get_derived_wrapper(gen: DirectCGenerator, helper: ModuleHelper) -> None:
    """Wrapper for derived-type scalar getters returning handles."""
    wrapper_name = module_helper_wrapper_name(helper)
    helper_symbol = module_helper_symbol(helper, gen.prefix)

    gen.write(
        f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
    )
    gen.write("{")
    gen.indent()
    gen.write("(void)self;")

    _write_derived_getter_body(gen, helper, helper_symbol)

    gen.write(f"PyObject* result = PyList_New({gen.handle_size});")
    gen.write("if (result == NULL) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"for (int i = 0; i < {gen.handle_size}; ++i) {{")
    gen.indent()
    gen.write("PyObject* item = PyLong_FromLong((long)value_handle[i]);")
    gen.write("if (item == NULL) {")
    gen.indent()
    gen.write("Py_DECREF(result);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("PyList_SET_ITEM(result, i, item);")
    gen.dedent()
    gen.write("}")
    gen.write("return result;")
    gen.dedent()
    gen.write("}")
    gen.write("")


def _write_derived_setter_args(gen: DirectCGenerator, helper: ModuleHelper) -> None:
    """Helper to parse arguments for derived setter."""
    gen.write("(void)self;")
    gen.write("PyObject* py_parent = Py_None;")
    gen.write("PyObject* py_value = Py_None;")

    if helper.is_type_member:
        gen.write("static char *kwlist[] = {\"handle\", \"value\", NULL};")
        gen.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"OO\", kwlist, &py_parent, &py_value)) {"
        )
    else:
        gen.write("static char *kwlist[] = {\"value\", NULL};")
        gen.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"O\", kwlist, &py_value)) {"
        )
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("")


def write_module_set_derived_wrapper(gen: DirectCGenerator, helper: ModuleHelper) -> None:
    """Wrapper for derived-type scalar setters accepting handles."""
    wrapper_name = module_helper_wrapper_name(helper)
    helper_symbol = module_helper_symbol(helper, gen.prefix)

    gen.write(
        f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
    )
    gen.write("{")
    gen.indent()

    _write_derived_setter_args(gen, helper)

    if helper.is_type_member:
        extract_parent_handle(gen, "parent")
        gen.write("Py_DECREF(parent_sequence);")
    else:
        gen.write(f"int parent_handle[{gen.handle_size}] = {{0}};")

    gen.write(f"int value_handle[{gen.handle_size}] = {{0}};")
    gen.write("PyObject* value_sequence = PySequence_Fast(py_value, \"Value must be a sequence\");")
    gen.write("if (value_sequence == NULL) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("Py_ssize_t value_len = PySequence_Fast_GET_SIZE(value_sequence);")
    gen.write(f"if (value_len != {gen.handle_size}) {{")
    gen.indent()
    gen.write("Py_DECREF(value_sequence);")
    gen.write("PyErr_SetString(PyExc_ValueError, \"Unexpected handle length\");")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write(f"for (int i = 0; i < {gen.handle_size}; ++i) {{")
    gen.indent()
    gen.write("PyObject* item = PySequence_Fast_GET_ITEM(value_sequence, i);")
    gen.write("value_handle[i] = (int)PyLong_AsLong(item);")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write("Py_DECREF(value_sequence);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("}")
    gen.write("Py_DECREF(value_sequence);")

    if helper.is_type_member:
        gen.write(f"{helper_symbol}(parent_handle, value_handle);")
    else:
        gen.write(f"{helper_symbol}(value_handle);")
    gen.write("Py_RETURN_NONE;")
    gen.dedent()
    gen.write("}")
    gen.write("")


def write_module_helper_declaration(gen: DirectCGenerator, helper: ModuleHelper) -> None:
    """Write extern declaration for module-level helper routines."""
    symbol = module_helper_symbol(helper, gen.prefix)
    if helper.kind in {"get", "set", "get_derived", "set_derived"}:
        c_type = c_type_from_fortran(helper.element.type, gen.kind_map)
        is_char = helper.element.type.strip().lower().startswith("character")
        if helper.kind in {"get_derived", "set_derived"}:
            if helper.is_type_member:
                gen.write(f"extern void {symbol}(int* handle, int* value);")
            else:
                gen.write(f"extern void {symbol}(int* value);")
        elif helper.is_type_member:
            if is_char:
                gen.write(
                    f"extern void {symbol}(int* handle, char* value, int value_len);"
                )
            else:
                gen.write(f"extern void {symbol}(int* handle, {c_type}* value);")
        else:
            if is_char:
                gen.write(f"extern void {symbol}(char* value, int value_len);")
            else:
                gen.write(f"extern void {symbol}({c_type}* value);")
    elif helper.kind == "array":
        # Module-level arrays do not need dummy_this argument (issue #306)
        if helper.is_type_member:
            gen.write(
                f"extern void {symbol}(int* dummy_this, int* nd, int* dtype, int* dshape, long long* handle);"
            )
        else:
            gen.write(
                f"extern void {symbol}(int* nd, int* dtype, int* dshape, long long* handle);"
            )
    elif helper.kind == "array_getitem":
        # Module-level arrays do not need dummy_this argument (issue #306)
        if helper.is_type_member:
            gen.write(
                f"extern void {symbol}(int* dummy_this, int* index, int* handle);"
            )
        else:
            gen.write(
                f"extern void {symbol}(int* index, int* handle);"
            )
    elif helper.kind == "array_setitem":
        # Module-level arrays do not need dummy_this argument (issue #306)
        if helper.is_type_member:
            gen.write(
                f"extern void {symbol}(int* dummy_this, int* index, int* handle);"
            )
        else:
            gen.write(
                f"extern void {symbol}(int* index, int* handle);"
            )
    elif helper.kind == "array_len":
        # Module-level arrays do not need dummy_this argument (issue #306)
        if helper.is_type_member:
            gen.write(
                f"extern void {symbol}(int* dummy_this, int* length);"
            )
        else:
            gen.write(
                f"extern void {symbol}(int* length);"
            )


def write_module_init(gen: DirectCGenerator, mod_name: str) -> None:
    """Generate module initialization function."""
    py_mod_name = gen.py_module_name if gen.py_module_name else mod_name
    gen.write(f"static struct PyModuleDef {mod_name}module = {{")
    gen.indent()
    gen.write("PyModuleDef_HEAD_INIT,")
    gen.write(f'"{py_mod_name}",')
    gen.write(f'"Direct-C wrapper for {mod_name} module",')
    gen.write("-1,")
    gen.write(f"{mod_name}_methods")
    gen.dedent()
    gen.write("};")
    gen.write("")

    gen.write(f"/* Module initialization */")
    gen.write(f"PyMODINIT_FUNC PyInit_{mod_name}(void)")
    gen.write("{")
    gen.indent()
    gen.write("import_array();  /* Initialize NumPy */")
    gen.write(f"PyObject* module = PyModule_Create(&{mod_name}module);")
    gen.write("if (module == NULL) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    if gen.callbacks:
        for callback_name in gen.callbacks:
            attr = callback_name
            gen.write("Py_INCREF(Py_None);")
            gen.write(
                f"if (PyModule_AddObject(module, \"{attr}\", Py_None) < 0) {{"
            )
            gen.indent()
            gen.write("Py_DECREF(Py_None);")
            gen.write("Py_DECREF(module);")
            gen.write("return NULL;")
            gen.dedent()
            gen.write("}")
    gen.write("return module;")
    gen.dedent()
    gen.write("}")
    alias_name = mod_name.lstrip("_")
    if alias_name and alias_name != mod_name and alias_name != py_mod_name:
        gen.write("")
        gen.write(f"PyMODINIT_FUNC PyInit_{py_mod_name}(void)")
        gen.write("{")
        gen.indent()
        gen.write(f"return PyInit_{mod_name}();")
        gen.dedent()
        gen.write("}")


from .derived_types import write_type_member_get_wrapper, write_type_member_set_wrapper
