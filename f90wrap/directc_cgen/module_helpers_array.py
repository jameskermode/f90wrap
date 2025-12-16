"""Array helper utilities for Direct-C module wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .utils import ModuleHelper, module_helper_wrapper_name, module_helper_symbol

if TYPE_CHECKING:
    from . import DirectCGenerator


def write_array_helper_body(gen: DirectCGenerator, helper: ModuleHelper, helper_symbol: str) -> None:
    """Emit the core body shared by array helpers."""
    # Module-level arrays do not need handle argument (issue #306)
    if helper.is_type_member:
        gen.write("PyObject* dummy_handle = Py_None;")
        gen.write("static char *kwlist[] = {\"handle\", NULL};")
        gen.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"|O\", kwlist, &dummy_handle)) {"
        )
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.write("")

        _write_handle_extraction(gen)
    else:
        gen.write("if (args && PyTuple_Size(args) != 0) {")
        gen.indent()
        gen.write("PyErr_SetString(PyExc_TypeError, \"Module-level array accessors do not take arguments\");")
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")

    gen.write("int nd = 0;")
    gen.write("int dtype = 0;")
    gen.write("int dshape[10] = {0};")
    gen.write("long long handle = 0;")
    if helper.is_type_member:
        gen.write(f"{helper_symbol}(dummy_this, &nd, &dtype, dshape, &handle);")
    else:
        gen.write(f"{helper_symbol}(&nd, &dtype, dshape, &handle);")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    _write_array_helper_return(gen)


def _write_handle_extraction(gen: DirectCGenerator) -> None:
    """Helper to extract handle from Python object."""
    gen.write("int dummy_this[4] = {0, 0, 0, 0};")
    gen.write("if (dummy_handle != Py_None) {")
    gen.indent()
    gen.write("PyObject* handle_sequence = PySequence_Fast(dummy_handle, \"Handle must be a sequence\");")
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
    gen.write(f"for (int i = 0; i < {gen.handle_size}; ++i) {{")
    gen.indent()
    gen.write("PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);")
    gen.write("if (item == NULL) {")
    gen.indent()
    gen.write("Py_DECREF(handle_sequence);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("dummy_this[i] = (int)PyLong_AsLong(item);")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write("Py_DECREF(handle_sequence);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("}")
    gen.write("Py_DECREF(handle_sequence);")
    gen.dedent()
    gen.write("}")


def _write_array_helper_return(gen: DirectCGenerator) -> None:
    """Helper to create array helper return tuple."""
    gen.write("if (nd < 0 || nd > 10) {")
    gen.indent()
    gen.write("PyErr_SetString(PyExc_ValueError, \"Invalid dimensionality\");")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write("PyObject* shape_tuple = PyTuple_New(nd);")
    gen.write("if (shape_tuple == NULL) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write("for (int i = 0; i < nd; ++i) {")
    gen.indent()
    gen.write("PyObject* dim = PyLong_FromLong((long)dshape[i]);")
    gen.write("if (dim == NULL) {")
    gen.indent()
    gen.write("Py_DECREF(shape_tuple);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("PyTuple_SET_ITEM(shape_tuple, i, dim);")
    gen.dedent()
    gen.write("}")

    gen.write("PyObject* result = PyTuple_New(4);")
    gen.write("if (result == NULL) {")
    gen.indent()
    gen.write("Py_DECREF(shape_tuple);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("PyTuple_SET_ITEM(result, 0, PyLong_FromLong((long)nd));")
    gen.write("PyTuple_SET_ITEM(result, 1, PyLong_FromLong((long)dtype));")
    gen.write("PyTuple_SET_ITEM(result, 2, shape_tuple);")
    gen.write("PyTuple_SET_ITEM(result, 3, PyLong_FromLongLong(handle));")
    gen.write("return result;")


def extract_parent_handle(gen: DirectCGenerator, parent_name: str = "parent") -> None:
    """Extract a parent handle sequence into a C array."""
    gen.write(
        f"PyObject* {parent_name}_sequence = PySequence_Fast(py_{parent_name}, \"Handle must be a sequence\");"
    )
    gen.write(f"if ({parent_name}_sequence == NULL) {{")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(
        f"Py_ssize_t {parent_name}_len = PySequence_Fast_GET_SIZE({parent_name}_sequence);"
    )
    gen.write(f"if ({parent_name}_len != {gen.handle_size}) {{")
    gen.indent()
    gen.write(f"Py_DECREF({parent_name}_sequence);")
    gen.write("PyErr_SetString(PyExc_ValueError, \"Unexpected handle length\");")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write(f"int {parent_name}_handle[{gen.handle_size}] = {{0}};")
    gen.write(f"for (int i = 0; i < {gen.handle_size}; ++i) {{")
    gen.indent()
    gen.write(f"PyObject* item = PySequence_Fast_GET_ITEM({parent_name}_sequence, i);")
    gen.write("if (item == NULL) {")
    gen.indent()
    gen.write(f"Py_DECREF({parent_name}_sequence);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"{parent_name}_handle[i] = (int)PyLong_AsLong(item);")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write(f"Py_DECREF({parent_name}_sequence);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("}")


def write_module_array_getitem_wrapper(gen: DirectCGenerator, helper: ModuleHelper) -> None:
    """Wrapper for module-level derived-type array getitem."""
    wrapper_name = module_helper_wrapper_name(helper)
    helper_symbol = module_helper_symbol(helper, gen.prefix)

    gen.write(
        f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
    )
    gen.write("{")
    gen.indent()
    gen.write("(void)self;")

    # Module-level arrays do not need handle argument (issue #306)
    if helper.is_type_member:
        gen.write("PyObject* py_parent;")
        gen.write("int index = 0;")
        gen.write("static char *kwlist[] = {\"handle\", \"index\", NULL};")
        gen.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"Oi\", kwlist, &py_parent, &index)) {"
        )
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")

        extract_parent_handle(gen, "parent")
    else:
        gen.write("int index = 0;")
        gen.write("static char *kwlist[] = {\"index\", NULL};")
        gen.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"i\", kwlist, &index)) {"
        )
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")

    gen.write(f"int handle[{gen.handle_size}] = {{0}};")
    if helper.is_type_member:
        gen.write(f"{helper_symbol}(parent_handle, &index, handle);")
        gen.write("if (PyErr_Occurred()) {")
        gen.indent()
        gen.write("Py_DECREF(parent_sequence);")
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.write("Py_DECREF(parent_sequence);")
    else:
        gen.write(f"{helper_symbol}(&index, handle);")
        gen.write("if (PyErr_Occurred()) {")
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")

    gen.write(f"PyObject* result = PyList_New({gen.handle_size});")
    gen.write("if (result == NULL) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"for (int i = 0; i < {gen.handle_size}; ++i) {{")
    gen.indent()
    gen.write("PyObject* item = PyLong_FromLong((long)handle[i]);")
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


def _extract_value_handle(gen: DirectCGenerator) -> None:
    """Helper to extract target value handle."""
    gen.write("PyObject* value_handle_obj = NULL;")
    gen.write("PyObject* value_sequence = NULL;")
    gen.write("Py_ssize_t value_handle_len = 0;")
    gen.write("if (PyObject_HasAttrString(py_value, \"_handle\")) {")
    gen.indent()
    gen.write("value_handle_obj = PyObject_GetAttrString(py_value, \"_handle\");")
    gen.write("if (value_handle_obj == NULL) { return NULL; }")
    gen.write(
        "value_sequence = PySequence_Fast(value_handle_obj, \"Failed to access handle sequence\");"
    )
    gen.write("if (value_sequence == NULL) { Py_DECREF(value_handle_obj); return NULL; }")
    gen.dedent()
    gen.write("} else if (PySequence_Check(py_value)) {")
    gen.indent()
    gen.write(
        "value_sequence = PySequence_Fast(py_value, \"Argument value must be a handle sequence\");"
    )
    gen.write("if (value_sequence == NULL) { return NULL; }")
    gen.dedent()
    gen.write("} else {")
    gen.indent()
    gen.write(
        "PyErr_SetString(PyExc_TypeError, \"Value must be a handle sequence or object with _handle\");"
    )
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write("value_handle_len = PySequence_Fast_GET_SIZE(value_sequence);")
    gen.write(f"if (value_handle_len != {gen.handle_size}) {{")
    gen.indent()
    gen.write("Py_DECREF(value_sequence);")
    gen.write("if (value_handle_obj) { Py_DECREF(value_handle_obj); }")
    gen.write("PyErr_SetString(PyExc_ValueError, \"Unexpected handle length\");")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write(f"int value_handle[{gen.handle_size}] = {{0}};")
    gen.write(f"for (int i = 0; i < {gen.handle_size}; ++i) {{")
    gen.indent()
    gen.write("PyObject* item = PySequence_Fast_GET_ITEM(value_sequence, i);")
    gen.write("if (item == NULL) {")
    gen.indent()
    gen.write("if (value_handle_obj) { Py_DECREF(value_handle_obj); }")
    gen.write("Py_DECREF(value_sequence);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("value_handle[i] = (int)PyLong_AsLong(item);")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write("if (value_handle_obj) { Py_DECREF(value_handle_obj); }")
    gen.write("Py_DECREF(value_sequence);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("}")
    gen.write("if (value_handle_obj) { Py_DECREF(value_handle_obj); }")


def write_module_array_setitem_wrapper(gen: DirectCGenerator, helper: ModuleHelper) -> None:
    """Wrapper for module-level derived-type array setitem."""
    wrapper_name = module_helper_wrapper_name(helper)
    helper_symbol = module_helper_symbol(helper, gen.prefix)

    gen.write(
        f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
    )
    gen.write("{")
    gen.indent()
    gen.write("(void)self;")

    # Module-level arrays do not need handle argument (issue #306)
    if helper.is_type_member:
        gen.write("PyObject* py_parent;")
        gen.write("int index = 0;")
        gen.write("PyObject* py_value;")
        gen.write("static char *kwlist[] = {\"handle\", \"index\", \"value\", NULL};")
        gen.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"OiO\", kwlist, &py_parent, &index, &py_value)) {"
        )
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")

        extract_parent_handle(gen, "parent")
        _extract_value_handle(gen)

        gen.write(f"{helper_symbol}(parent_handle, &index, value_handle);")
        gen.write("Py_DECREF(parent_sequence);")
        gen.write("if (value_sequence) { Py_DECREF(value_sequence); }")
    else:
        gen.write("int index = 0;")
        gen.write("PyObject* py_value;")
        gen.write("static char *kwlist[] = {\"index\", \"value\", NULL};")
        gen.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"iO\", kwlist, &index, &py_value)) {"
        )
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")

        _extract_value_handle(gen)

        gen.write(f"{helper_symbol}(&index, value_handle);")
        gen.write("if (value_sequence) { Py_DECREF(value_sequence); }")

    gen.write("Py_RETURN_NONE;")
    gen.dedent()
    gen.write("}")
    gen.write("")


def write_module_array_len_wrapper(gen: DirectCGenerator, helper: ModuleHelper) -> None:
    """Wrapper returning the length of derived-type arrays."""
    wrapper_name = module_helper_wrapper_name(helper)
    helper_symbol = module_helper_symbol(helper, gen.prefix)

    gen.write(
        f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
    )
    gen.write("{")
    gen.indent()
    gen.write("(void)self;")

    # Module-level arrays do not need handle argument (issue #306)
    if helper.is_type_member:
        gen.write("PyObject* py_parent;")
        gen.write("static char *kwlist[] = {\"handle\", NULL};")
        gen.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"O\", kwlist, &py_parent)) {"
        )
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")

        extract_parent_handle(gen, "parent")

        gen.write("int length = 0;")
        gen.write(f"{helper_symbol}(parent_handle, &length);")
        gen.write("Py_DECREF(parent_sequence);")
    else:
        gen.write("if (args && PyTuple_Size(args) != 0) {")
        gen.indent()
        gen.write("PyErr_SetString(PyExc_TypeError, \"Module-level array length does not take arguments\");")
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")

        gen.write("int length = 0;")
        gen.write(f"{helper_symbol}(&length);")
    gen.write("return PyLong_FromLong((long)length);")
    gen.dedent()
    gen.write("}")
    gen.write("")
