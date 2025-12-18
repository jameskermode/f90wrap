"""Direct-C C code generator for f90wrap."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from f90wrap import codegen as cg
from f90wrap import fortran as ft
from f90wrap.directc import InteropInfo, ProcedureKey
from f90wrap.transform import shorten_long_name
from f90wrap.numpy_utils import c_type_from_fortran

from .utils import (
    ModuleHelper,
    helper_name,
    helper_symbol,
    module_helper_name,
    module_helper_symbol,
    module_helper_wrapper_name,
    wrapper_name,
    build_value_map,
    is_hidden_argument,
)
from .arguments import write_arg_parsing, write_arg_preparation
from .module_helpers import (
    write_module_helper_wrapper,
    write_module_helper_declaration,
    write_module_init,
)
from .procedures import (
    write_wrapper_function,
    write_helper_declaration,
    write_alias_helper_declaration,
)


@dataclass
class DirectCGenerator(cg.CodeGenerator):
    """Generate C extension module code calling f90wrap helpers."""

    root: ft.Root
    interop_info: Dict[ProcedureKey, InteropInfo]
    kind_map: Dict[str, Dict[str, str]]
    prefix: str = "f90wrap_"
    handle_size: int = 4
    error_num_arg: Optional[str] = None
    error_msg_arg: Optional[str] = None
    callbacks: Optional[Iterable[str]] = None
    shape_hints: Optional[Dict[Tuple[str, Optional[str], str, str], List[str]]] = None
    py_module_name: Optional[str] = None

    def __post_init__(self):
        """Initialize CodeGenerator parent after dataclass init."""
        cg.CodeGenerator.__init__(self, indent="    ", max_length=120,
                                   continuation="\\", comment="//")
        if self.callbacks:
            seen = set()
            ordered: List[str] = []
            for name in self.callbacks:
                if not name:
                    continue
                normalised = name.strip()
                if not normalised:
                    continue
                key = normalised.lower()
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(normalised)
            self.callbacks = ordered
        else:
            self.callbacks = []
        if self.shape_hints is None:
            self.shape_hints = {}

    def generate_module(
        self,
        mod_name: str,
        procedures: Optional[Iterable[ft.Procedure]] = None,
    ) -> str:
        """Generate complete _module.c file content."""

        # Reset code buffer
        self.code = []
        self._write_headers(mod_name)

        module_label = mod_name.lstrip("_") or mod_name

        selected = self._collect_procedures(module_label, procedures)
        module_helpers = self._collect_module_helpers(module_label, procedures)
        binding_aliases = self._collect_binding_aliases(module_label)

        if not selected and not module_helpers and not binding_aliases:
            has_candidates = bool(procedures)
            if not has_candidates:
                return ""

        self._write_external_declarations(selected, module_helpers, binding_aliases)

        # Generate wrapper functions
        for proc in selected:
            write_wrapper_function(self, proc, mod_name)

        for helper in module_helpers:
            write_module_helper_wrapper(self, helper)

        alias_wrappers: List[Tuple[str, str, ft.Binding]] = []
        for alias_name, binding, proc in binding_aliases:
            wrapper_name = self._write_binding_alias_wrapper(alias_name, binding, proc, mod_name)
            alias_wrappers.append((alias_name, wrapper_name, binding))

        # Module method table and init
        self._write_method_table(selected, module_helpers, alias_wrappers, mod_name)
        write_module_init(self, mod_name)

        return str(self)

    def _collect_procedures(
        self,
        mod_name: str,
        procedures: Optional[Iterable[ft.Procedure]] = None,
    ) -> List[ft.Procedure]:
        """Collect procedures that require helper wrappers for a module."""

        if procedures is None:
            proc_list: List[ft.Procedure] = []
        else:
            proc_list = list(procedures)

        target_module: Optional[ft.Module] = None
        for module in self.root.modules:
            if module.name == mod_name:
                target_module = module
                if procedures is None:
                    module_procs = getattr(module, "procedures", [])
                    try:
                        module_iter = list(module_procs)
                    except TypeError:
                        module_iter = []
                    proc_list.extend(module_iter)
                break

        if target_module is not None:
            types_attr = getattr(target_module, "types", [])
            try:
                types_iter = list(types_attr)
            except TypeError:
                types_iter = []
            for derived in types_iter:
                procs_attr = getattr(derived, "procedures", [])
                try:
                    derived_procs = list(procs_attr)
                except TypeError:
                    derived_procs = []
                is_abstract = getattr(derived, "abstract", False)
                for proc in derived_procs:
                    if proc not in proc_list:
                        # Skip abstract procedures (including init/finalise for abstract types)
                        proc_attrs = getattr(proc, "attributes", [])
                        if "abstract" in proc_attrs:
                            continue
                        # Also skip init/finalise for abstract types - they cannot be instantiated
                        if is_abstract and proc.name.endswith(("_initialise", "_finalise")):
                            continue
                        proc_list.append(proc)

        # Filter out abstract procedures from the final list
        selected: List[ft.Procedure] = []
        for proc in proc_list:
            proc_attrs = getattr(proc, "attributes", [])
            if "abstract" in proc_attrs:
                continue
            selected.append(proc)

        return selected

    def _collect_module_elements(self, module: ft.Module) -> List[ModuleHelper]:
        """Collect helpers for module-level elements."""
        helpers: List[ModuleHelper] = []
        seen: set[Tuple[str, str, str, bool]] = set()

        elements_attr = getattr(module, "elements", [])
        try:
            elements_iter = list(elements_attr)
        except TypeError:
            elements_iter = []

        # Build types dictionary for checking has_assignment
        types_by_name = self._get_types_by_name()

        for element in elements_iter:
            is_array = any(attr.startswith("dimension(") for attr in element.attributes)
            is_parameter = any(attr.startswith("parameter") for attr in element.attributes)
            element_type = element.type.strip().lower()
            is_derived = element_type.startswith("type(") or element_type.startswith("class(")

            if not is_array:
                if is_derived:
                    key = (module.name, element.name, "get_derived", False)
                    if key not in seen:
                        helpers.append(ModuleHelper(module.name, element.name, "get_derived", element, False))
                        seen.add(key)
                    if not is_parameter and not self._should_skip_setter(element, types_by_name):
                        key = (module.name, element.name, "set_derived", False)
                        if key not in seen:
                            helpers.append(ModuleHelper(module.name, element.name, "set_derived", element, False))
                            seen.add(key)
                else:
                    key = (module.name, element.name, "get", False)
                    if key not in seen:
                        helpers.append(ModuleHelper(module.name, element.name, "get", element, False))
                        seen.add(key)
                    if not is_parameter:
                        key = (module.name, element.name, "set", False)
                        if key not in seen:
                            helpers.append(ModuleHelper(module.name, element.name, "set", element, False))
                            seen.add(key)
            elif is_derived:
                for kind in ("array_getitem", "array_setitem", "array_len"):
                    if kind == "array_setitem" and is_parameter:
                        continue
                    if kind == "array_setitem" and self._should_skip_setter(element, types_by_name):
                        continue
                    key = (module.name, element.name, kind, False)
                    if key not in seen:
                        helpers.append(ModuleHelper(module.name, element.name, kind, element, False))
                        seen.add(key)
            else:
                key = (module.name, element.name, "array", False)
                if key not in seen:
                    helpers.append(ModuleHelper(module.name, element.name, "array", element, False))
                    seen.add(key)

        return helpers

    def _get_types_by_name(self) -> Dict[str, ft.Type]:
        """Build dictionary mapping type names to Type objects."""
        types_by_name: Dict[str, ft.Type] = {}
        for mod in self.root.modules:
            types_attr = getattr(mod, "types", [])
            try:
                types_list = list(types_attr)
            except TypeError:
                types_list = []
            for t in types_list:
                types_by_name[ft.strip_type(t.name)] = t
        return types_by_name

    def _should_skip_setter(self, element: ft.Element, types_by_name: Dict[str, ft.Type]) -> bool:
        """Check if setter should be skipped for polymorphic types without assignment."""
        # Polymorphic objects require an assignment(=) method to be set
        if not ft.is_class(element.type):
            return False

        type_name = ft.strip_type(element.type)
        type_obj = types_by_name.get(type_name)
        if type_obj is None:
            return False

        attributes = getattr(type_obj, "attributes", [])
        return "has_assignment" not in attributes

    def _collect_type_elements(self, module: ft.Module, derived: ft.Type) -> List[ModuleHelper]:
        """Collect helpers for derived type elements."""
        helpers: List[ModuleHelper] = []
        seen: set[Tuple[str, str, str, bool]] = set()

        module_scope = (getattr(module, "orig_name", None) or module.name).lower()
        type_mod = f"{module_scope}__{derived.name}"
        elements_attr = getattr(derived, "elements", [])
        try:
            derived_elements = list(elements_attr)
        except TypeError:
            derived_elements = []

        # Build types dictionary for checking has_assignment
        types_by_name = self._get_types_by_name()

        for element in derived_elements:
            is_array = any(attr.startswith("dimension(") for attr in element.attributes)
            is_parameter = any(attr.startswith("parameter") for attr in element.attributes)
            element_type = element.type.strip().lower()
            is_derived = element_type.startswith("type(") or element_type.startswith("class(")

            if not is_array:
                if is_derived:
                    key = (type_mod, element.name, "get_derived", True)
                    if key not in seen:
                        helpers.append(ModuleHelper(type_mod, element.name, "get_derived", element, True))
                        seen.add(key)
                    if not is_parameter and not self._should_skip_setter(element, types_by_name):
                        key = (type_mod, element.name, "set_derived", True)
                        if key not in seen:
                            helpers.append(ModuleHelper(type_mod, element.name, "set_derived", element, True))
                            seen.add(key)
                else:
                    key = (type_mod, element.name, "get", True)
                    if key not in seen:
                        helpers.append(ModuleHelper(type_mod, element.name, "get", element, True))
                        seen.add(key)
                    if not is_parameter:
                        key = (type_mod, element.name, "set", True)
                        if key not in seen:
                            helpers.append(ModuleHelper(type_mod, element.name, "set", element, True))
                            seen.add(key)
                continue

            if is_derived:
                for kind in ("array_getitem", "array_setitem", "array_len"):
                    if kind == "array_setitem" and is_parameter:
                        continue
                    if kind == "array_setitem" and self._should_skip_setter(element, types_by_name):
                        continue
                    key = (type_mod, element.name, kind, True)
                    if key not in seen:
                        helpers.append(ModuleHelper(type_mod, element.name, kind, element, True))
                        seen.add(key)
            else:
                key = (type_mod, element.name, "array", True)
                if key not in seen:
                    helpers.append(ModuleHelper(type_mod, element.name, "array", element, True))
                    seen.add(key)

        return helpers

    def _collect_module_helpers(
        self, mod_name: str, procedures: Optional[Iterable[ft.Procedure]] = None
    ) -> List[ModuleHelper]:
        """Collect helper metadata for module-level variables."""
        helpers: List[ModuleHelper] = []

        for module in self.root.modules:
            # Collect module elements
            helpers.extend(self._collect_module_elements(module))

            # Collect type elements
            types_attr = getattr(module, "types", [])
            try:
                types_iter = list(types_attr)
            except TypeError:
                types_iter = []

            for derived in types_iter:
                helpers.extend(self._collect_type_elements(module, derived))

        return helpers

    def _write_headers(self, module_name: str) -> None:
        """Write standard C headers and Python/NumPy includes."""

        self.write("#include <Python.h>")
        self.write("#include <stdbool.h>")
        self.write("#include <stdlib.h>")
        self.write("#include <string.h>")
        self.write("#include <complex.h>")
        self.write("")
        self.write("#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION")
        self.write("/* Use same array API symbol as numpy f2py for fortranobject.c compatibility */")
        self.write("#define PY_ARRAY_UNIQUE_SYMBOL _npy_f2py_ARRAY_API")
        self.write("#include <numpy/arrayobject.h>")
        self.write("")
        self.write("#define F90WRAP_F_SYMBOL(name) name##_")
        self.write("")
        self._write_abort_helpers()
        target = self.py_module_name if self.py_module_name else module_name
        self._write_callback_trampolines(module_name, target)

    def _write_abort_helpers(self) -> None:
        """Emit minimal abort handler expected by f90wrap helpers."""

        self.write("void f90wrap_abort_(char *message, int len_message)")
        self.write("{")
        self.indent()
        self.write("/* Acquire GIL since we're calling Python C-API from Fortran */")
        self.write("PyGILState_STATE gstate = PyGILState_Ensure();")
        self.write("")
        self.write("if (message == NULL) {")
        self.indent()
        self.write("PyErr_SetString(PyExc_RuntimeError, \"f90wrap_abort called\");")
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("while (len_message > 0 && message[len_message - 1] == ' ') {")
        self.indent()
        self.write("--len_message;")
        self.dedent()
        self.write("}")
        self.write("if (len_message <= 0) {")
        self.indent()
        self.write("PyErr_SetString(PyExc_RuntimeError, \"f90wrap_abort called\");")
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("PyObject* unicode = PyUnicode_FromStringAndSize(message, len_message);")
        self.write("if (unicode == NULL) {")
        self.indent()
        self.write("PyErr_SetString(PyExc_RuntimeError, \"f90wrap_abort called\");")
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("PyErr_SetObject(PyExc_RuntimeError, unicode);")
        self.write("Py_DECREF(unicode);")
        self.write("PyGILState_Release(gstate);")
        self.dedent()
        self.write("}")
        self.write("")
        self.write("void f90wrap_abort__(char *message, int len_message)")
        self.write("{")
        self.indent()
        self.write("f90wrap_abort_(message, len_message);")
        self.dedent()
        self.write("}")
        self.write("")

    def _write_callback_trampolines(self, module_name: str, wrapper_name: str) -> None:
        """Emit helper and trampoline functions for registered callbacks."""

        if not self.callbacks:
            return

        self.write("/* Callback trampolines */")
        self.write(
            "static void f90wrap_invoke_callback(const char *callback_name, char *message, int len_message)"
        )
        self.write("{")
        self.indent()
        self.write("PyGILState_STATE gstate = PyGILState_Ensure();")
        self.write("")
        self.write("/* Validate input buffer and length */")
        self.write("if (message == NULL) {")
        self.indent()
        self.write("PyErr_SetString(PyExc_RuntimeError, \"callback message is NULL\");")
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("if (len_message < 0) {")
        self.indent()
        self.write("len_message = 0;")
        self.dedent()
        self.write("}")
        self.write("")
        self.write(f"PyObject* module = PyImport_AddModule(\"{module_name}\");")
        self.write("if (module == NULL) {")
        self.indent()
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("PyObject* callable = PyObject_GetAttrString(module, callback_name);")
        self.write("if (callable == NULL || callable == Py_None) {")
        self.indent()
        self.write("PyErr_Clear();")
        self.write("Py_XDECREF(callable);")
        self.write(f"PyObject* wrapper = PyImport_AddModule(\"{wrapper_name}\");")
        self.write("if (wrapper != NULL) {")
        self.indent()
        self.write("callable = PyObject_GetAttrString(wrapper, callback_name);")
        self.dedent()
        self.write("} else {")
        self.indent()
        self.write("callable = NULL;")
        self.dedent()
        self.write("}")
        self.dedent()
        self.write("}")
        self.write("if (callable == NULL || callable == Py_None) {")
        self.indent()
        self.write("Py_XDECREF(callable);")
        self.write("PyErr_Format(PyExc_RuntimeError, \"cb: Callback %s not defined (as an argument or module attribute).\", callback_name);")
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("int actual_len = len_message;")
        self.write("while (actual_len > 0 && message[actual_len - 1] == ' ') {")
        self.indent()
        self.write("--actual_len;")
        self.dedent()
        self.write("}")
        self.write("PyObject* arg_bytes = PyBytes_FromStringAndSize(message, actual_len);")
        self.write("if (arg_bytes == NULL) {")
        self.indent()
        self.write("Py_DECREF(callable);")
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("PyObject* arg = PyUnicode_DecodeLatin1(PyBytes_AS_STRING(arg_bytes), actual_len, \"strict\");")
        self.write("Py_DECREF(arg_bytes);")
        self.write("if (arg == NULL) {")
        self.indent()
        self.write("Py_DECREF(callable);")
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("PyObject* result = PyObject_CallFunctionObjArgs(callable, arg, NULL);")
        self.write("Py_DECREF(arg);")
        self.write("Py_DECREF(callable);")
        self.write("if (result == NULL) {")
        self.indent()
        self.write("/* Report exception to aid debugging, then clear it */")
        self.write("PyObject* callback_str = PyUnicode_FromString(callback_name);")
        self.write("if (callback_str != NULL) {")
        self.indent()
        self.write("PyErr_WriteUnraisable(callback_str);")
        self.write("Py_DECREF(callback_str);")
        self.dedent()
        self.write("}")
        self.write("PyErr_Clear();")
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("Py_DECREF(result);")
        self.write("PyGILState_Release(gstate);")
        self.dedent()
        self.write("}")
        self.write("")

        for callback_name in self.callbacks:
            symbol = f"{callback_name.lower()}_"
            self.write(f"void {symbol}(char *message, int len_message)")
            self.write("{")
            self.indent()
            self.write(f"f90wrap_invoke_callback(\"{callback_name}\", message, len_message);")
            self.dedent()
            self.write("}")
            self.write("")

    def _write_external_declarations(
        self,
        procedures: Iterable[ft.Procedure],
        module_helpers: Iterable[ModuleHelper],
        binding_aliases: Iterable[Tuple[str, ft.Binding, ft.Procedure]],
    ) -> None:
        """Write extern declarations for f90wrap helper functions."""

        self.write("/* External f90wrap helper functions */")
        for proc in procedures:
            write_helper_declaration(self, proc)
        for helper in module_helpers:
            write_module_helper_declaration(self, helper)
        for alias_name, binding, proc in binding_aliases:
            write_alias_helper_declaration(self, alias_name, binding, proc)
        self.write("")

    def _write_binding_alias_wrapper(
        self,
        alias_name: str,
        binding: ft.Binding,
        proc: ft.Procedure,
        mod_name: str,
    ) -> str:
        """Generate a wrapper that forwards to a binding-specific helper."""

        prev_value_map = getattr(self, "_value_map", None)
        prev_hidden = getattr(self, "_hidden_names", set())
        prev_hidden_lower = getattr(self, "_hidden_names_lower", set())
        prev_proc = getattr(self, "_current_proc", None)

        self._value_map = build_value_map(proc)
        self._hidden_names = {arg.name for arg in proc.arguments if is_hidden_argument(arg)}
        self._hidden_names_lower = {name.lower() for name in self._hidden_names}
        self._current_proc = proc

        suffix = alias_name[len(self.prefix):] if alias_name.startswith(self.prefix) else alias_name
        func_wrapper_name = shorten_long_name(f"wrap__{suffix}")

        self.write(
            f"static PyObject* {func_wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
        )
        self.write("{")
        self.indent()
        self.write("(void)self;")

        write_arg_parsing(self, proc)
        write_arg_preparation(self, proc)

        self.write("/* Call f90wrap helper */")
        helper_sym = helper_symbol(proc, self.prefix)
        from .procedures import write_helper_call, write_return_value
        write_helper_call(self, proc, helper_sym=helper_sym)

        write_return_value(self, proc)

        self.dedent()
        self.write("}")
        self.write("")

        self._value_map = prev_value_map
        self._hidden_names = prev_hidden
        self._hidden_names_lower = prev_hidden_lower
        self._current_proc = prev_proc

        return func_wrapper_name

    def _get_all_bindings(self, derived_type: ft.Type, types_by_name: Dict[str, ft.Type]) -> List[ft.Binding]:
        """Get all bindings including those inherited from parent types."""
        all_bindings = []

        # Get bindings from this type
        bindings_attr = getattr(derived_type, "bindings", [])
        try:
            own_bindings = list(bindings_attr)
        except TypeError:
            own_bindings = []
        all_bindings.extend(own_bindings)

        # Get bindings from parent type recursively
        parent = getattr(derived_type, "parent", None)
        if parent:
            # parent could be a Type object or a string
            if isinstance(parent, ft.Type):
                parent_name = parent.name
            else:
                parent_name = str(parent)

            parent_type = types_by_name.get(parent_name)
            if parent_type:
                parent_bindings = self._get_all_bindings(parent_type, types_by_name)
                # Only add parent bindings that aren't overridden
                own_binding_names = {b.name for b in own_bindings}
                for pb in parent_bindings:
                    if pb.name not in own_binding_names:
                        all_bindings.append(pb)

        return all_bindings

    def _collect_binding_aliases(self, mod_name: str) -> List[Tuple[str, ft.Binding, ft.Procedure]]:
        """Collect alias names for type-bound procedures from all modules."""

        aliases: List[Tuple[str, ft.Binding, ft.Procedure]] = []

        # Build a GLOBAL map of type names to types for cross-module inheritance lookup
        global_types_by_name: Dict[str, ft.Type] = {}
        for mod in self.root.modules:
            types_attr = getattr(mod, "types", [])
            try:
                types_list = list(types_attr)
            except TypeError:
                types_list = []
            for t in types_list:
                global_types_by_name[t.name] = t

        # Iterate through ALL modules in the tree, not just the one matching mod_name
        # since mod_name is the extension name, not the Fortran module name
        for module in self.root.modules:
            module_procs = getattr(module, "procedures", [])
            try:
                module_proc_list = list(module_procs)
            except TypeError:
                module_proc_list = []
            procedures_by_name = {proc.name: proc for proc in module_proc_list}

            types_attr = getattr(module, "types", [])
            try:
                types_iter = list(types_attr)
            except TypeError:
                types_iter = []

            derived_proc_map: Dict[str, List[ft.Procedure]] = {}
            for derived in types_iter:
                procs_attr = getattr(derived, "procedures", [])
                try:
                    derived_procs = list(procs_attr)
                except TypeError:
                    derived_procs = []
                derived_proc_map[derived.name] = derived_procs
                for proc in derived_procs:
                    if proc.name not in procedures_by_name:
                        procedures_by_name[proc.name] = proc

            for derived in types_iter:
                # Get all bindings including inherited ones (using global type map for cross-module inheritance)
                all_bindings = self._get_all_bindings(derived, global_types_by_name)

                for binding in all_bindings:
                    if binding.type not in ("procedure", "final"):
                        continue
                    # Note: deferred bindings ARE wrapped - the Fortran wrapper uses
                    # runtime polymorphism to dispatch to the correct implementation
                    targets = getattr(binding, "procedures", [])
                    if not targets:
                        continue
                    target = targets[0]
                    proc: Optional[ft.Procedure] = None
                    if isinstance(target, ft.Procedure):
                        proc = target
                    else:
                        target_name = getattr(target, "name", None)
                        if not target_name:
                            continue
                        proc = procedures_by_name.get(target_name)
                        if proc is None:
                            for candidate in derived_proc_map.get(derived.name, []):
                                if candidate.name == target_name:
                                    proc = candidate
                                    break
                    if proc is None:
                        continue
                    # Use procedure's module name for alias (important for inherited bindings)
                    # For inherited bindings, the procedure may be from a parent module
                    proc_mod_name = getattr(proc, 'mod_name', None) or module.name

                    # For polymorphic bindings, the Fortran wrapper name includes parent type
                    # e.g. f90wrap_m_base_poly__is_polygone__binding__polygone_rectangle
                    # For multi-level inheritance, it includes the full chain:
                    # e.g. f90wrap_m_base_poly__is_polygone__binding__polygone_rectangle_square
                    proc_type_name = getattr(proc, 'type_name', None)
                    if proc_type_name and proc_type_name.lower() != derived.name.lower():
                        # Binding is from a parent type - build the inheritance chain
                        # from the binding's original type down to this derived type
                        type_chain = [derived.name.lower()]
                        current = derived
                        while current:
                            parent = getattr(current, 'parent', None)
                            if not parent:
                                break
                            if isinstance(parent, ft.Type):
                                parent_name = parent.name
                            else:
                                parent_name = str(parent)

                            # Stop when we reach the type that owns this binding
                            if parent_name.lower() == proc_type_name.lower():
                                type_chain.insert(0, parent_name.lower())
                                break

                            # Add parent to chain and continue up
                            type_chain.insert(0, parent_name.lower())
                            current = global_types_by_name.get(parent_name)

                        # Join chain with underscores
                        type_suffix = "_".join(type_chain)
                        alias = shorten_long_name(
                            f"f90wrap_{proc_mod_name}__{binding.name}__binding__{type_suffix}"
                        )
                    else:
                        # Binding is from this type - just use derivedtype
                        alias = shorten_long_name(
                            f"f90wrap_{proc_mod_name}__{binding.name}__binding__{derived.name.lower()}"
                        )
                    aliases.append((alias, binding, proc))

        return aliases

    def _write_method_table(
        self,
        procedures: List[ft.Procedure],
        module_helpers: List[ModuleHelper],
        alias_wrappers: List[Tuple[str, str, ft.Binding]],
        mod_name: str,
    ) -> None:
        """Write the module method table."""

        self.write(f"/* Method table for {mod_name} module */")
        self.write(f"static PyMethodDef {mod_name}_methods[] = {{")
        self.indent()

        for proc in procedures:
            func_wrapper_name = wrapper_name(mod_name, proc)
            method_name = helper_name(proc, self.prefix)
            docstring = proc.doc[0] if proc.doc else f"Wrapper for {proc.name}"
            # Escape any quotes and newlines in docstring
            docstring = docstring.replace('"', '\\"').replace('\n', '\\n')
            self.write(
                f'{{"{method_name}", (PyCFunction){func_wrapper_name}, '
                f"METH_VARARGS | METH_KEYWORDS, \"{docstring}\"}},"
            )

        for helper in module_helpers:
            func_wrapper_name = module_helper_wrapper_name(helper)
            method_name = module_helper_name(helper, self.prefix)
            if helper.kind == "array":
                docstring = f"Array helper for {helper.name}"
            else:
                docstring = f"Module helper for {helper.name}"
            docstring = docstring.replace('"', '\\"').replace('\n', '\\n')
            self.write(
                f'{{"{method_name}", (PyCFunction){func_wrapper_name}, '
                f"METH_VARARGS | METH_KEYWORDS, \"{docstring}\"}},"
            )

        for alias_name, func_wrapper_name, binding in alias_wrappers:
            docstring = f"Binding alias for {binding.name}"
            docstring = docstring.replace('"', '\\"').replace('\n', '\\n')
            self.write(
                f'{{"{alias_name}", (PyCFunction){func_wrapper_name}, '
                f"METH_VARARGS | METH_KEYWORDS, \"{docstring}\"}},"
            )

        self.write("{NULL, NULL, 0, NULL}  /* Sentinel */")
        self.dedent()
        self.write("};")
        self.write("")
