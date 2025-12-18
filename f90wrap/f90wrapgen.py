#  f90wrap: F90 to Python interface generator with derived type support
#
#  Copyright James Kermode 2011-2018
#
#  This file is part of f90wrap
#  For the latest version see github.com/jameskermode/f90wrap
#
#  f90wrap is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  f90wrap is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with f90wrap. If not, see <http://www.gnu.org/licenses/>.
#
#  If you would like to license the source code under different terms,
#  please contact James Kermode, james.kermode@gmail.com

import logging
import os
import warnings
import re

import numpy as np
from typing import Set

from f90wrap import codegen as cg
from f90wrap import fortran as ft
from f90wrap import directc
from f90wrap.transform import shorten_long_name

log = logging.getLogger(__name__)


class F90WrapperGenerator(ft.FortranVisitor, cg.CodeGenerator):
    """
    Creates the Fortran90 code necessary to wrap a given Fortran parse tree
    suitable for input to `f2py`.

    Each node of the tree (Module, Subroutine etc.) is wrapped according to the
    rules in this class when visited (using `F90WrapperGenerator.visit()`).

    Each module's wrapper is written to a separate file, with top-level
    procedures written to another separate file. Derived-types and arrays (both
    of normal types and derive-types) are specially treated. For each, a number
    of subroutines allowing the getting/setting of items, and retrieval of array
    length are written. Furthermore, derived-types are treated as opaque
    references to enable wrapping with `f2py`.

    Parameters
    ----------
    prefix : `str`
        A string with which to prefix module, subroutine and type names.

    sizeof_fortran_t : `int`
        The size, in bytes, of a pointer to a fortran derived type ??

    string_lengths : `dict`
        This is never used...

    abort_func : `str`
        Name of a Fortran function to be invoked when a fatal error occurs

    kind_map : `dict`
        Dictionary mapping Fortran types and kinds to C-types

    types: `dict`
        Dictionary mapping type names to Fortran modules where they are defined
    """

    def __init__(
        self,
        prefix,
        sizeof_fortran_t,
        string_lengths,
        abort_func,
        kind_map,
        types,
        default_to_inout,
        max_length=None,
        auto_raise=None,
        default_string_length=None,
        direct_c_interop=None,
        toplevel_basename="toplevel",
    ):
        if max_length is None:
            max_length = 120
        cg.CodeGenerator.__init__(
            self, indent=" " * 4, max_length=max_length, continuation="&", comment="!"
        )
        ft.FortranVisitor.__init__(self)
        self.prefix = prefix
        self.sizeof_fortran_t = sizeof_fortran_t
        self.string_lengths = string_lengths
        self.abort_func = abort_func
        self.kind_map = kind_map
        self.types = types
        self.default_to_inout = default_to_inout
        self.routines: Set[tuple] = set()
        try:
            self._err_num_var, self._err_msg_var = auto_raise.split(',')
        except ValueError:
            self._err_num_var, self._err_msg_var = None, None
        self.default_string_length = default_string_length
        self.direct_c_interop = direct_c_interop or {}
        self.toplevel_basename = toplevel_basename
        self._namespace_helper = directc.NamespaceHelper(
            self.types,
            namespace_types=bool(self.direct_c_interop),
        )

    def _scope_identifier_for(self, container):
        """Build a stable identifier used to namespace generated helper names."""
        return self._namespace_helper.scope_identifier_for(container)

    def _register_modules(self, root):
        """Cache module nodes by both generated and original names."""
        self._namespace_helper.register_modules(root)

    def _find_type(self, type_name, module_hint=None):
        """Locate a Type node, preferring the provided module hint."""
        return self._namespace_helper.find_type(type_name, module_hint)

    def _type_owner(self, type_name, module_hint=None):
        """Return the defining module name for a given type."""
        return self._namespace_helper.type_owner(type_name, module_hint)

    def _add_extra_use(self, extra_uses, module_name, symbol):
        """Append a symbol to a module's ONLY list, avoiding duplicates."""
        self._namespace_helper.add_extra_use(extra_uses, module_name, symbol)

    def _direct_c_info(self, proc):
        if not self.direct_c_interop:
            return None
        if not hasattr(proc, "mod_name") or not hasattr(proc, "name"):
            return None
        key = directc.ProcedureKey(proc.mod_name, getattr(proc, 'type_name', None), proc.name)
        return self.direct_c_interop.get(key)

    def visit_Root(self, node):
        """
        Write a wrapper for top-level procedures.
        """
        self._register_modules(node)
        # clean up any previous wrapper files
        top_level_wrapper_file = "%s%s.f90" % (
            self.prefix,
            self.toplevel_basename,
        )
        f90_wrapper_files = [
            "%s%s.f90"
            % (self.prefix, os.path.splitext(os.path.basename(mod.filename))[0])
            for mod in node.modules
        ] + [top_level_wrapper_file]

        for f90_wrapper_file in f90_wrapper_files:
            if os.path.exists(f90_wrapper_file):
                os.unlink(f90_wrapper_file)
        self.code = []
        self.generic_visit(node)
        if len(self.code) > 0:
            f90_wrapper_file = open(top_level_wrapper_file, "w")
            f90_wrapper_file.write(str(self))
            f90_wrapper_file.close()

    def visit_Module(self, node):
        """
        Wrap modules. Each Fortran module generates one wrapper source file.

        Subroutines and elements within each module are properly wrapped.
        """
        if hasattr(node, "is_external") and node.is_external:
            return None
        log.info("F90WrapperGenerator visiting module %s" % node.name)
        self.code = []
        self.write("! Module %s defined in file %s" % (node.name, node.filename))
        self.write()
        self.generic_visit(node)

        for el in node.elements:
            dims = list(filter(lambda x: x.startswith("dimension"), el.attributes))
            if len(dims) == 0:  # proper scalar type (normal or derived)
                self._write_scalar_wrappers(node, el, self.sizeof_fortran_t)
            elif ft.is_derived_type(el.type):  # array of derived types
                self._write_dt_array_wrapper(node, el, dims[0], self.sizeof_fortran_t)
            else:
                if "parameter" not in el.attributes:
                    self._write_sc_array_wrapper(
                        node, el, dims[0], self.sizeof_fortran_t
                    )

        self.write("! End of module %s defined in file %s" % (node.name, node.filename))
        self.write()
        if len(self.code) > 0:
            f90_wrapper_name = "%s%s.f90" % (
                self.prefix,
                os.path.splitext(os.path.basename(node.filename))[0],
            )
            if os.path.exists(f90_wrapper_name):
                warnings.warn(
                    "Source file %s contains code for more than one module!"
                    % node.filename
                )
            f90_wrapper_file = open(f90_wrapper_name, "a")
            f90_wrapper_file.write(str(self))
            f90_wrapper_file.close()
        self.code = []

    def write_uses_lines(self, node, extra_uses_dict=None):
        """
        Write "uses mod, only: sub" lines to the code.

        Parameters
        ----------
        node : Node of parse tree
        """
        all_uses = {}
        node_module = getattr(node, "mod_name", None)
        if node_module:
            self._add_extra_use(all_uses, node_module, None)
        if hasattr(node, "uses"):
            for use in node.uses:
                if isinstance(use, str):
                    self._add_extra_use(all_uses, use, None)
                else:
                    mod, only = use
                    if only is None:
                        self._add_extra_use(all_uses, mod, None)
                    else:
                        for symbol in only:
                            self._add_extra_use(all_uses, mod, symbol)

        if extra_uses_dict is not None:
            for mod, info in extra_uses_dict.items():
                if isinstance(info, dict):
                    if info.get("full"):
                        self._add_extra_use(all_uses, mod, None)
                    for symbol in info.get("symbols", []):
                        self._add_extra_use(all_uses, mod, symbol)
                elif info is None:
                    self._add_extra_use(all_uses, mod, None)
                else:
                    for symbol in info:
                        self._add_extra_use(all_uses, mod, symbol)

        if (
            hasattr(node, "attributes")
            and "destructor" in node.attributes
            and not "skip_call" in node.attributes
        ):
            self._add_extra_use(all_uses, node.mod_name, node.call_name)

        if all_uses:
            symbol_sources = {}
            for mod, entry in list(all_uses.items()):
                if entry["full"]:
                    continue
                cleaned = []
                for symbol in entry["symbols"]:
                    symbol_str = symbol if isinstance(symbol, str) else str(symbol)
                    if "=>" in symbol_str:
                        cleaned.append(symbol_str)
                        continue
                    base_symbol = symbol_str.strip()
                    owner = self._type_owner(base_symbol)
                    if owner and owner != mod:
                        if symbol_sources.get(base_symbol) is not None:
                            continue
                    prev = symbol_sources.get(base_symbol)
                    if prev and prev != mod:
                        if owner and prev == owner:
                            continue
                        if owner and mod != owner:
                            continue
                    symbol_sources[base_symbol] = mod
                    cleaned.append(base_symbol)
                if cleaned:
                    entry["symbols"] = cleaned
                else:
                    if entry["full"]:
                        entry["symbols"] = []
                    else:
                        del all_uses[mod]

        for mod, entry in all_uses.items():
            if entry["full"]:
                self.write(f"use {mod}")
        for mod, entry in all_uses.items():
            symbols = entry["symbols"]
            if not symbols:
                continue
            ordered = list(dict.fromkeys(symbols))
            self.write(f"use {mod}, only: {', '.join(ordered)}")

    def write_super_type_lines(self, ty):
        self.write("type " + ty.name)
        self.indent()
        for el in ty.elements:
            self.write(
                el.type
                + "".join(", " + attr for attr in el.attributes)
                + " :: "
                + el.name
            )
        self.dedent()
        self.write("end type " + ty.name)
        self.write()

    def write_type_lines(self, tname, recursive=False, tname_inner=None, *, pointer=False):
        """
        Write a pointer type for a given type name

        Parameters
        ----------
        tname : `str`
            Should be the name of a derived type in the wrapped code.

        recursive : `boolean`
            Adjusts array pointer for recursive derived type array
        """
        tname = ft.strip_type(tname)
        if tname_inner is None:
            tname_inner = tname

        type_node = self._find_type(tname)
        if type_node is None:
            type_node = self.types.get(ft.strip_type(tname))
        attributes = getattr(type_node, "attributes", []) if type_node is not None else []
        if "abstract" in attributes:
            class_type = "class"
        else:
            class_type = "type"

        if not recursive:
            suffix = "_ptr_type"
        else:
            suffix = "_rec_ptr_type"

        self.write(
            """type %(typename)s%(suffix)s
    %(class_type)s(%(typename_inner)s), pointer :: p => NULL()
end type %(typename)s%(suffix)s"""
            % {"suffix": suffix, "class_type": class_type, "typename": tname, "typename_inner": tname_inner}
        )

    def write_class_lines(self, cname, recursive=False, *, pointer=False):
        """
        Write a pointer type for a given class name

        Parameters
        ----------
        tname : `str`
            Should be the name of a class in the wrapped code.
        """
        pointer_str = "pointer :: obj => NULL()" if pointer else "allocatable  :: obj"
        cname = ft.strip_type(cname)
        self.write(
            "type %(classname)s_wrapper_type\n"
            "    class(%(classname)s), %(pointer_str)s\n"
            "end type %(classname)s_wrapper_type" % {"classname": cname, "pointer_str": pointer_str}
        )
        self.write_type_lines(cname, recursive, f"{cname}_wrapper_type")

    def is_class(self, tname):
        if not tname:
            return False
        type_node = self._find_type(tname)
        if type_node is None:
            type_node = self.types.get(ft.strip_type(tname))
        if type_node is None:
            return False
        attributes = getattr(type_node, "attributes", [])
        if "used_as_class" in attributes:
            return True
        fallback = self.types.get(ft.strip_type(tname))
        if fallback is not None and "used_as_class" in getattr(fallback, "attributes", []):
            return True
        if "has_assignment" in getattr(fallback, "attributes", []):
            return True
        return False

    def write_type_or_class_lines(self, tname, recursive=False, *, pointer=False):
        if self.is_class(tname):
            self.write_class_lines(tname, recursive, pointer=pointer)
        else:
            self.write_type_lines(tname, recursive, pointer=pointer)


    def write_arg_decl_lines(self, node, helper_forward=False):
        """
        Write argument declaration lines to the code

        Takes care of argument attributes, and opaque references for derived
        types, as well as f2py-specific lines.
        """
        direct_c_integer_scalars = []

        for arg in node.arguments:
            if "callback" in arg.attributes:
                return "external " + arg.name

            attributes = [
                attr
                for attr in arg.attributes
                if attr
                in ("optional", "pointer", "intent(in)", "intent(out)", "intent(inout)")
                or attr.startswith("dimension")
            ]
            arg_dict = {
                "arg_type": arg.type,
                "type_name": (arg.type.startswith("type") and arg.type[5:-1])
                or (arg.type.startswith("class") and arg.type[6:-1])
                or None,
                "arg_name": arg.name,
            }  # self.prefix+arg.name}

            if arg.name in node.transfer_in or arg.name in node.transfer_out:
                if not helper_forward:
                    self.write(
                        "type(%(type_name)s_ptr_type) :: %(arg_name)s_ptr" % arg_dict
                    )
                arg_dict["arg_type"] = arg.wrapper_type
                attributes.append("dimension(%d)" % arg.wrapper_dim)

            if (
                self._err_num_var and self._err_msg_var
                and arg_dict["arg_name"] in [self._err_num_var, self._err_msg_var]
            ):
                attributes = []
                if arg_dict["arg_name"] == "errmsg":
                    arg_dict["arg_type"] = "character*(%s)" % self.default_string_length
            arg_dict["arg_attribs"] = ", ".join(attributes)
            arg_dict["comma"] = len(attributes) != 0 and ", " or ""

            if self.direct_c_interop:
                arg_type = str(arg_dict["arg_type"]).strip().lower()
                is_integer_scalar = arg_type == "integer" and not any(
                    str(attr).startswith("dimension(") for attr in getattr(arg, "attributes", [])
                )
                if is_integer_scalar and "optional" not in arg.attributes:
                    arg_dict["arg_type"] = "integer(c_int)"
                    direct_c_integer_scalars.append(arg.name)

            # character array definition
            # https://github.com/numpy/numpy/issues/18684
            if arg.type == "character(*)":
                arg_dict["arg_type"] = "character*(*)"
            self.write(
                "%(arg_type)s%(comma)s%(arg_attribs)s :: %(arg_name)s" % arg_dict
            )
            if hasattr(arg, "f2py_line"):
                self.write(arg.f2py_line)
            elif self.default_to_inout and all(
                "intent" not in attr for attr in arg.attributes
            ):
                # No f2py instruction and no explicit intent : force f2py to make the argument intent(inout)
                # This is put as an option to prserve backwards compatibility
                self.write("!f2py intent(inout) " + arg.name)

        if direct_c_integer_scalars:
            for name in direct_c_integer_scalars:
                self.write(f"integer :: {self.prefix}{name}_default")

        # For allocatable class returns, declare temporary variable for move_alloc
        orig_node = getattr(node, 'orig_node', node)
        if isinstance(orig_node, ft.Function):
            is_allocatable_return = any("allocatable" in attr.lower() for attr in orig_node.ret_val.attributes)
            if is_allocatable_return:
                is_class_return = self.is_class(orig_node.ret_val.type)
                # Only declare temp for classes (which use wrapper_type and can use move_alloc)
                # For non-class derived types, we can't use move_alloc with pointer target
                if is_class_return:
                    type_name = (orig_node.ret_val.type.startswith("type") and orig_node.ret_val.type[5:-1]) or \
                               (orig_node.ret_val.type.startswith("class") and orig_node.ret_val.type[6:-1])
                    temp_name = f"temp_{orig_node.ret_val.name}"
                    self.write(f"class({type_name}), allocatable :: {temp_name}")

    def write_transfer_in_lines(self, node):
        """
        Write transfer of opaque references.
        """
        for arg in node.arguments:
            arg_dict = {
                "arg_name": arg.name,  # self.prefix+arg.name,
                "arg_type": arg.type,
            }
            if arg.name in node.transfer_in:
                if "optional" in arg.attributes:
                    self.write("if (present(%(arg_name)s)) then" % arg_dict)
                    self.indent()

                self.write(
                    "%(arg_name)s_ptr = transfer(%(arg_name)s, %(arg_name)s_ptr)"
                    % arg_dict
                )

                if "optional" in arg.attributes:
                    self.dedent()
                    self.write("else")
                    self.indent()
                    if self.is_class(arg.type):
                        node.deallocate.append(arg.name)
                        self.write("allocate(%(arg_name)s_ptr%%p)" % arg_dict)
                        self.write("%(arg_name)s_ptr%%p%%obj => null()" % arg_dict)
                    else:
                        self.write("%(arg_name)s_ptr%%p => null()" % arg_dict)
                    self.dedent()
                    self.write("end if")

    def write_init_lines(self, node):
        """
        Write special user-provided init lines to a node.
        """
        for alloc in node.allocate:
            # For return values that are allocatable, don't pre-allocate the pointer
            # The assignment will handle allocation automatically (Fortran 2003 semantics)
            is_allocatable_return = any("allocatable" in attr.lower() for attr in alloc.attributes)
            is_class_return = self.is_class(alloc.type)

            if not is_allocatable_return:
                # Non-allocatable returns: allocate both pointer and object
                self.write("allocate(%s_ptr%%p)" % alloc.name)
                if is_class_return:
                    self.write("allocate(%s_ptr%%p%%obj)" % alloc.name)
            elif is_class_return:
                # Allocatable class returns: allocate pointer wrapper but not the object
                # The move_alloc will move the allocatable result into p%obj
                self.write("allocate(%s_ptr%%p)" % alloc.name)
        for arg in node.arguments:
            if not hasattr(arg, "init_lines"):
                continue
            exe_optional, exe = arg.init_lines
            D = {
                "OLD_ARG": arg.name,
                "ARG": arg.name,  # self.prefix+arg.name,
                "PTR": arg.name + "_ptr%p",
            }
            if "optional" in arg.attributes:
                self.write(exe_optional % D)
            else:
                self.write(exe % D)

    def write_call_lines(self, node, func_name, helper_forward=False):
        """
        Write line that calls a single wrapped Fortran routine
        """
        if "skip_call" in node.attributes:
            return

        orig_node = node
        arg_node = node
        if hasattr(node, "orig_node"):
            orig_node = node.orig_node
            arg_node = orig_node  # get arguemnt list from original node

        def dummy_arg_name(arg):
            return arg.orig_name

        def is_type_a_class(arg_type):
            if arg_type.startswith("class") and arg_type[6:-1]:
                return True
            if arg_type.startswith("type") and arg_type[5:-1]:
                tname = arg_type[5:-1]
                if self.is_class(tname):
                    return True
            return False

        def actual_arg_name(arg):
            if helper_forward:
                return arg.name
            name = arg.name
            if self.direct_c_interop:
                arg_type = str(getattr(arg, "type", "")).strip().lower()
                is_integer_scalar = arg_type == "integer" and not any(
                    str(attr).startswith("dimension(") for attr in getattr(arg, "attributes", [])
                )
                if is_integer_scalar and "optional" not in getattr(arg, "attributes", []):
                    return f"{self.prefix}{arg.name}_default"
            if (hasattr(node, "transfer_in") and arg.name in node.transfer_in) or (
                hasattr(node, "transfer_out") and arg.name in node.transfer_out
            ):
                name += "_ptr%p"
            if is_type_a_class(arg.type):
                name += "%obj"
            if "super-type" in arg.doc:
                name += "%items"
            return name

        if node.mod_name is not None:
            # use keyword arguments if subroutine is in a module and we have an explicit interface
            arg_names = [
                "%s=%s" % (dummy_arg_name(arg), actual_arg_name(arg))
                for arg in arg_node.arguments
                if "intent(hide)" not in arg.attributes
            ]
        else:
            arg_names = [
                actual_arg_name(arg)
                for arg in arg_node.arguments
                if "intent(hide)" not in arg.attributes
            ]

        # If procedure is bound make the call via type%bound_name syntax
        bound_name = None
        for attr in node.attributes:
            match = re.match(r"bound\((.*?)\)", attr)
            if match:
                bound_name = match.group(1)
        if bound_name:
            # Remove this in argument passing
            call_name = None
            for arg in arg_names[:]:
                if not call_name:
                    match = re.search("=(.*_ptr)", arg)
                    if match:
                        call_name = match.group(1)
                        arg_names.remove(arg)

            if (self.is_class(node.type_name)):
                func_name = "%s%%p%%obj%%%s" % (call_name, bound_name)
            else:
                func_name = "%s%%p%%%s" % (call_name, bound_name)

        has_err_vars = False
        if self._err_num_var is not None and self._err_msg_var is not None:
            dummy_names = [
                dummy_arg_name(arg)
                for arg in arg_node.arguments
                if "intent(hide)" not in arg.attributes
            ]
            has_err_vars = (
                self._err_num_var in dummy_names and self._err_msg_var in dummy_names
            )
            if has_err_vars:
                self.write(f"{self._err_num_var}=0")
                self.write(f"{self._err_msg_var}=''")

        direct_c_integer_scalars = []
        if self.direct_c_interop:
            for arg in node.arguments:
                arg_type = str(getattr(arg, "type", "")).strip().lower()
                is_integer_scalar = arg_type == "integer" and not any(
                    str(attr).startswith("dimension(") for attr in getattr(arg, "attributes", [])
                )
                if is_integer_scalar and "optional" not in getattr(arg, "attributes", []):
                    direct_c_integer_scalars.append(arg)

        for arg in direct_c_integer_scalars:
            intent = "inout"
            attrs = getattr(arg, "attributes", [])
            if any(str(a).startswith("intent(in)") for a in attrs):
                intent = "in"
            if any(str(a).startswith("intent(out)") for a in attrs):
                intent = "out"
            if any(str(a).startswith("intent(inout)") for a in attrs):
                intent = "inout"
            if intent in {"in", "inout"}:
                self.write(f"{self.prefix}{arg.name}_default = {arg.name}")

        if isinstance(orig_node, ft.Function):
            # For allocatable returns, use move_alloc or allocate+assign appropriately
            ret_val_name = actual_arg_name(orig_node.ret_val)
            is_allocatable_return = any("allocatable" in attr.lower() for attr in orig_node.ret_val.attributes)
            is_class_return = self.is_class(orig_node.ret_val.type)
            is_derived_type = orig_node.ret_val.type.startswith("type(")

            if is_allocatable_return:
                if is_class_return:
                    # Use temporary + move_alloc to avoid finalizer calls on assignment
                    temp_name = f"temp_{orig_node.ret_val.name}"
                    self.write(
                        "%(temp_name)s = %(func_name)s(%(arg_names)s)"
                        % {
                            "temp_name": temp_name,
                            "func_name": func_name,
                            "arg_names": ", ".join(arg_names),
                        }
                    )
                    self.write(f"call move_alloc({temp_name}, {ret_val_name})")
                elif is_derived_type:
                    # For allocatable derived type returns without wrapper, use allocate with source=
                    # This copies the function result and automatically deallocates the temporary
                    self.write(
                        "allocate(%(ret_val)s, source=%(func_name)s(%(arg_names)s))"
                        % {
                            "ret_val": ret_val_name,
                            "func_name": func_name,
                            "arg_names": ", ".join(arg_names),
                        }
                    )
                else:
                    # For allocatable intrinsic types, use move_alloc
                    temp_name = f"temp_{orig_node.ret_val.name}"
                    self.write(
                        "%(temp_name)s = %(func_name)s(%(arg_names)s)"
                        % {
                            "temp_name": temp_name,
                            "func_name": func_name,
                            "arg_names": ", ".join(arg_names),
                        }
                    )
                    self.write(f"call move_alloc({temp_name}, {ret_val_name})")
            else:
                self.write(
                    "%(ret_val)s = %(func_name)s(%(arg_names)s)"
                    % {
                        "ret_val": ret_val_name,
                        "func_name": func_name,
                        "arg_names": ", ".join(arg_names),
                    }
                )
        else:
            if func_name == "assignment(=)":
                if len(arg_names) != 2:
                    raise RuntimeError(
                        "assignment(=) interface with len(arg_names) != 2"
                    )
                arg_names = [arg_name.split("=")[1] for arg_name in arg_names]
                self.write(
                    "%(lhs)s = %(rhs)s" % {"lhs": arg_names[0], "rhs": arg_names[1]}
                )
            else:
                self.write(
                    "call %(sub_name)s(%(arg_names)s)"
                    % {"sub_name": func_name, "arg_names": ", ".join(arg_names)}
                )

        for arg in direct_c_integer_scalars:
            intent = "inout"
            attrs = getattr(arg, "attributes", [])
            if any(str(a).startswith("intent(in)") for a in attrs):
                intent = "in"
            if any(str(a).startswith("intent(out)") for a in attrs):
                intent = "out"
            if any(str(a).startswith("intent(inout)") for a in attrs):
                intent = "inout"
            if intent in {"out", "inout"}:
                self.write(f"{arg.name} = {self.prefix}{arg.name}_default")

        if has_err_vars:
            self.write(f"if ({self._err_num_var}.ne.0) then")
            self.indent()
            self.write(f"call {self.abort_func}({self._err_msg_var})")
            self.dedent()
            self.write("end if")

    def write_transfer_out_lines(self, node):
        """
        Write transfer from opaque reference.
        """
        for arg in node.arguments:
            if arg.name in node.transfer_out:
                self.write(
                    "%(arg_name)s = transfer(%(arg_name)s_ptr, %(arg_name)s)"
                    % {"arg_name": arg.name}
                )

    def write_finalise_lines(self, node):
        """
        Deallocate the opaque reference to clean up.
        """
        for dealloc in node.deallocate:
            is_optional = False
            is_class = False
            for arg in node.arguments:
                if ft.strip_type(arg.name) == dealloc and "optional" in arg.attributes:
                    is_optional = True
                if self.is_class(arg.type):
                    is_class = True
            if is_optional:
                self.write(f"if (.not. present({dealloc})) then")
                self.indent()
            if is_class and not is_optional:
                self.write(f"deallocate({dealloc}_ptr%p%obj)")
            self.write(f"deallocate({dealloc}_ptr%p)")
            if is_optional:
                self.dedent()
                self.write("end if")

    def visit_Procedure(self, node):
        """
        Write wrapper code necessary for a Fortran subroutine or function
        """
        if "abstract" in node.attributes and not "method" in node.attributes:
            return self.generic_visit(node)

        call_name = node.orig_name
        if hasattr(node, "call_name"):
            call_name = node.call_name

        type_name = getattr(node, "type_name", None)
        routine_key = (node.name, type_name, node.mod_name if type_name is None else None)
        if routine_key in self.routines:
            return self.generic_visit(node)

        self.routines.add(routine_key)

        log.info(
            "F90WrapperGenerator visiting routine %s call_name %s mod_name %r"
            % (node.name, call_name, node.mod_name)
        )

        filtered_arguments = node.arguments
        if self._err_num_var is not None and self._err_msg_var is not None:
            filtered_arguments = [
                arg for arg in filtered_arguments if arg.name not in [self._err_num_var, self._err_msg_var]
            ]

        sub_name = self.prefix + node.name
        arg_names = (
            "(" + ", ".join([arg.name for arg in filtered_arguments]) + ")"
            if filtered_arguments
            else ""
        )
        if node.mod_name is not None:
            sub_name = self.prefix + node.mod_name + '__' + node.name
        sub_name = shorten_long_name(sub_name)
        self.write("subroutine %s%s" % (sub_name, arg_names))
        self.indent()
        self.write_uses_lines(node)

        needs_c_int = False
        if self.direct_c_interop:
            for arg in node.arguments:
                arg_type = str(getattr(arg, "type", "")).strip().lower()
                is_integer_scalar = arg_type == "integer" and not any(
                    str(attr).startswith("dimension(") for attr in getattr(arg, "attributes", [])
                )
                if is_integer_scalar:
                    needs_c_int = True
                    break

        # Add iso_c_binding when we emit c_int declarations (handles and integer scalars).
        if node.transfer_in or node.transfer_out or needs_c_int:
            self.write("use, intrinsic :: iso_c_binding, only: c_int")
        self.write("implicit none")

        if node.mod_name is None:
            self.write("external %s" % call_name)
            if hasattr(node, "orig_node") and isinstance(node.orig_node, ft.Function):
                self.write("%s %s" % (node.orig_node.ret_val.type, node.orig_name))

        helper_forward = call_name.startswith(self.prefix)

        saved_types = None
        if helper_forward:
            saved_types = node.types
            node.types = set()

        self.write()
        if not helper_forward:
            for tname in node.types:
                type_node = self._find_type(tname, node.mod_name)
                if type_node is not None and "super-type" in getattr(type_node, "doc", ""):
                    self.write_super_type_lines(type_node)
                pointer = False
                for arg in node.arguments:
                    if ft.strip_type(arg.type) == tname and "optional" in arg.attributes:
                        pointer = True
                self.write_type_or_class_lines(tname, pointer=pointer)
        self.write_arg_decl_lines(node, helper_forward=helper_forward)
        if not helper_forward:
            self.write_transfer_in_lines(node)
        self.write_init_lines(node)
        self.write_call_lines(node, call_name, helper_forward=helper_forward)
        if not helper_forward:
            self.write_transfer_out_lines(node)
        self.write_finalise_lines(node)
        if helper_forward:
            node.types = saved_types
        self.dedent()
        self.write("end subroutine %s" % (sub_name))
        self.write()
        return self.generic_visit(node)

    def visit_Type(self, node):
        """
        Properly wraps derived types, including derived-type arrays.
        """
        log.info("F90WrapperGenerator visiting type %s" % node.name)

        for el in node.elements:
            dims = list(filter(lambda x: x.startswith("dimension"), el.attributes))
            if len(dims) == 0:  # proper scalar type (normal or derived)
                self._write_scalar_wrappers(node, el, self.sizeof_fortran_t)
            elif el.type.startswith(("type", "class")):  # array of derived types
                self._write_dt_array_wrapper(node, el, dims[0], self.sizeof_fortran_t)
            else:
                self._write_sc_array_wrapper(node, el, dims[0], self.sizeof_fortran_t)

        return self.generic_visit(node)

    def _get_type_member_array_name(self, t, element_name):
        if (self.is_class(t.orig_name)):
            return "this_ptr%%p%%obj%%%s" % element_name
        return "this_ptr%%p%%%s" % element_name

    def _write_sc_array_wrapper(self, t, el, dims, sizeof_fortran_t):
        """
        Write wrapper for arrays of intrinsic types

        Parameters
        ----------
        t : `fortran.Type` node
            Derived-type node of the parse tree.

        el : `fortran.Element` node
            An element of a module which is derived-type array

        dims : `tuple` of `int`s
            The dimensions of the element

        sizeof_fortan_t : `int`
            The size, in bytes, of a pointer to a fortran derived type ??

        """
        if isinstance(t, ft.Type):
            this = "this, "
        else:
            # Module-level arrays do not need a this argument since there is no
            # instance handle to pass.  Removing it avoids sizeof_fortran_t
            # mismatches between generation and runtime environments (issue #306).
            this = ""

        subroutine_name = "%s%s__array__%s" % (
            self.prefix,
            self._scope_identifier_for(t),
            el.name,
        )
        subroutine_name = shorten_long_name(subroutine_name)

        self.write("subroutine %s(%snd, dtype, dshape, dloc)" % (subroutine_name, this))
        self.indent()

        if isinstance(t, ft.Module):
            module_name = t.orig_name or t.name
            local_name = shorten_long_name("%s_%s" % (t.name, el.name))
            alias = "%s => %s" % (local_name, el.orig_name)
            extra_uses = {}
            self._add_extra_use(extra_uses, module_name, None)
            self._add_extra_use(extra_uses, module_name, alias)
            self.write_uses_lines(t, extra_uses)
        else:
            self.write_uses_lines(t)

        self.write("use, intrinsic :: iso_c_binding, only : c_int")
        self.write("implicit none")
        if isinstance(t, ft.Type):
            self.write_type_or_class_lines(t.orig_name)
            self.write("integer(c_int), intent(in) :: this(%d)" % sizeof_fortran_t)
            self.write("type(%s_ptr_type) :: this_ptr" % t.orig_name)

        self.write("integer(c_int), intent(out) :: nd")
        self.write("integer(c_int), intent(out) :: dtype")
        try:
            rank = len(ft.Argument.split_dimensions(dims))
            if el.type.startswith("character"):
                rank += 1
        except ValueError:
            rank = 1
        self.write("integer(c_int), dimension(10), intent(out) :: dshape")
        self.write("integer*%d, intent(out) :: dloc" % np.dtype("O").itemsize)
        self.write()
        self.write("nd = %d" % rank)
        self.write("dtype = %s" % ft.fortran_array_type(el.type, self.kind_map))
        if isinstance(t, ft.Type):
            self.write("this_ptr = transfer(this, this_ptr)")
            array_name = self._get_type_member_array_name(t, el.orig_name)
        else:
            array_name = shorten_long_name("%s_%s" % (t.name, el.name))

        pointer_guard = None
        if "allocatable" in el.attributes:
            pointer_guard = f"allocated({array_name})"
        elif "pointer" in el.attributes:
            pointer_guard = f"associated({array_name})"

        if pointer_guard is not None:
            self.write(f"if ({pointer_guard}) then")
            self.indent()
        if el.type.startswith("character"):
            first = ",".join(["1" for i in range(rank - 1)])
            self.write(
                "dshape(1:%d) = (/len(%s(%s)), shape(%s)/)"
                % (rank, array_name, first, array_name)
            )
        else:
            self.write("dshape(1:%d) = shape(%s)" % (rank, array_name))
        self.write("dloc = loc(%s)" % array_name)
        if pointer_guard is not None:
            self.dedent()
            self.write("else")
            self.indent()
            self.write(f"dshape(1:{rank}) = 0")
            self.write("dloc = 0")
            self.dedent()
            self.write("end if")

        self.dedent()
        self.write("end subroutine %s" % (subroutine_name))
        self.write()

    def _write_dt_array_wrapper(self, t, element, dims, sizeof_fortran_t):
        """
        Write fortran get/set/len routines for a (1-dimensional) derived-type array.

        Parameters
        ----------
        t : `fortran.Type` node
            Derived-type node of the parse tree.

        el : `fortran.Element` node
            An element of a module which is derived-type array

        dims : `tuple` of `int`s
            The dimensions of the element

        sizeof_fortan_t : `int`
            The size, in bytes, of a pointer to a fortran derived type ??
        """
        if (
            element.type.startswith(("type", "class"))
            and len(ft.Argument.split_dimensions(dims)) != 1
        ):
            return

        self._write_array_getset_item(t, element, sizeof_fortran_t, "get")
        # For CLASS arrays without has_assignment: only generate setitem in Direct-C mode
        # (Direct-C uses select type for polymorphic assignment; f2py can't parse it)
        if self.direct_c_interop or not ft.is_class(element.type) or "has_assignment" in self.types[ft.strip_type(element.type)].attributes:
            self._write_array_getset_item(t, element, sizeof_fortran_t, "set")
        self._write_array_len(t, element, sizeof_fortran_t)

    def _write_scalar_wrappers(self, t, element, sizeof_fortran_t):
        """
        Write fortran get/set routines for scalar derived-types

        Parameters
        ----------
        t : `fortran.Type` node
            Derived-type node of the parse tree.

        el : `fortran.Element` node
            An element of a module which is derived-type array

        sizeof_fortan_t : `int`
            The size, in bytes, of a pointer to a fortran derived type ??
        """
        self._write_scalar_wrapper(t, element, sizeof_fortran_t, "get")
        # Parameters cannot be set
        if "parameter" in element.attributes: return
        # Polymorphic objects require an assignment(=) method to be set
        if ft.is_class(element.type) and "has_assignment" not in self.types[ft.strip_type(element.type)].attributes:
            return None
        self._write_scalar_wrapper(t, element, sizeof_fortran_t, "set")

    def _write_array_getset_item(self, t, el, sizeof_fortran_t, getset):
        """
        Write a subroutine to get/set items in a derived-type array.

        Parameters
        ----------
        t : `fortran.Type` node
            Derived-type node of the parse tree.

        el : `fortran.Element` node
            An element of a module which is derived-type array

        sizeof_fortan_t : `int`
            The size, in bytes, of a pointer to a fortran derived type ??

        getset : `str` {``"get"``,``"set"``}
            String indicating whether to write a get routine, or a set routine.
        """
        # getset and inout just change things simply from a get to a set routine.
        inout = "in"
        if getset == "get":
            inout = "out"

        if isinstance(t, ft.Type):
            this = self.prefix + "this"
        else:
            # Module-level arrays do not need a this argument (issue #306).
            this = None
        safe_i = self.prefix + "i"  # YANN: i could be in the "uses" clauses
        # TODO: check if el.orig_name would be needed here instead of el.name
        subroutine_name = "%s%s__array_%sitem__%s" % (
            self.prefix,
            self._scope_identifier_for(t),
            getset,
            el.name,
        )
        subroutine_name = shorten_long_name(subroutine_name)

        if this is not None:
            self.write(
                "subroutine %s(%s, %s, %s)"
                % (subroutine_name, this, safe_i, el.name + "item")
            )
        else:
            self.write(
                "subroutine %s(%s, %s)"
                % (subroutine_name, safe_i, el.name + "item")
            )
        self.indent()
        self.write()
        extra_uses = {}
        if isinstance(t, ft.Module):
            local_name = shorten_long_name("%s_%s" % (t.name, el.name))
            alias = "%s => %s" % (local_name, el.orig_name)
            module_name = t.orig_name or t.name
            self._add_extra_use(extra_uses, module_name, None)
            self._add_extra_use(extra_uses, module_name, alias)
        elif isinstance(t, ft.Type):
            if "super-type" in t.doc:
                # YANN: propagate parameter uses
                for use in t.uses:
                    module_name = use[0]
                    only_list = use[1] if len(use) > 1 else []
                    for symbol in only_list:
                        self._add_extra_use(extra_uses, module_name, symbol)
            else:
                self._add_extra_use(extra_uses, t.mod_name, None)
        owner_module = self._type_owner(el.type, getattr(t, "mod_name", getattr(t, "name", None)))
        self._add_extra_use(extra_uses, owner_module, None)
        self.write_uses_lines(el, extra_uses)
        # Add iso_c_binding for handle arrays (c_int is immune to -fdefault-integer-8)
        self.write("use, intrinsic :: iso_c_binding, only: c_int")
        self.write("implicit none")
        self.write()

        if "super-type" in t.doc:
            self.write_super_type_lines(t)

        # Check if the type has recursive definition:
        same_type = ft.strip_type(t.name) == ft.strip_type(el.type)

        if isinstance(t, ft.Type):
            self.write_type_or_class_lines(t.name)
        self.write_type_or_class_lines(el.type, same_type, pointer=True)

        if this is not None:
            self.write("integer(c_int), intent(in) :: %s(%d)" % (this, sizeof_fortran_t))
        if isinstance(t, ft.Type):
            self.write("type(%s_ptr_type) :: this_ptr" % t.name)
            array_name = self._get_type_member_array_name(t, el.name)
        else:
            array_name = shorten_long_name("%s_%s" % (t.name, el.name))
        self.write("integer(c_int), intent(in) :: %s" % (safe_i))
        self.write(
            "integer(c_int), intent(%s) :: %s(%d)"
            % (inout, el.name + "item", sizeof_fortran_t)
        )
        if not same_type:
            self.write(
                "type(%s_ptr_type) :: %s_ptr" % (ft.strip_type(el.type), el.name)
            )
        else:
            self.write(
                "type(%s_rec_ptr_type) :: %s_ptr" % (ft.strip_type(el.type), el.name)
            )
        self.write()
        if isinstance(t, ft.Type):
            self.write("this_ptr = transfer(%s, this_ptr)" % (this))

        if "allocatable" in el.attributes:
            self.write("if (allocated(%s)) then" % array_name)
            self.indent()

        self.write("if (%s < 1 .or. %s > size(%s)) then" % (safe_i, safe_i, array_name))
        self.indent()
        self.write('call %s("array index out of range")' % self.abort_func)
        self.dedent()
        self.write("else")
        self.indent()

        if getset == "get":
            self.write("allocate(%s_ptr%%p)" % (el.name))
            if (self.is_class(el.type)):
                self.write("%s_ptr%%p%%obj => %s(%s)" % (el.name, array_name, safe_i))
            else:
                self.write("%s_ptr%%p => %s(%s)" % (el.name, array_name, safe_i))
            self.write(
                "%s = transfer(%s_ptr,%s)"
                % (el.name + "item", el.name, el.name + "item")
            )
        else:
            self.write(
                "%s_ptr = transfer(%s,%s_ptr)" % (el.name, el.name + "item", el.name)
            )
            if self.direct_c_interop and el.type.startswith("class"):
                # Direct-C mode: use select type for polymorphic assignment
                # (f2py doesn't understand select type syntax, so skip this for f2py)
                base_type = ft.strip_type(el.type)
                self.write("select type (lhs => %s(%s))" % (array_name, safe_i))
                self.indent()
                self.write("type is (%s)" % base_type)
                self.indent()
                self.write("select type (rhs => %s_ptr%%p%%obj)" % el.name)
                self.indent()
                self.write("type is (%s)" % base_type)
                self.indent()
                self.write("lhs = rhs")
                self.dedent()
                self.write("class default")
                self.indent()
                self.write('call %s("class array setitem type mismatch")' % self.abort_func)
                self.dedent()
                self.write("end select")
                self.dedent()
                self.dedent()
                self.write("class default")
                self.indent()
                self.write('call %s("class array setitem type mismatch")' % self.abort_func)
                self.dedent()
                self.write("end select")
                self.dedent()
            elif self.is_class(el.type):
                self.write(
                    "%s(%s) = %s_ptr%%p%%obj" % (array_name, safe_i, el.name)
                )
            else:
                self.write("%s(%s) = %s_ptr%%p" % (array_name, safe_i, el.name))

        self.dedent()
        self.write("endif")

        if "allocatable" in el.attributes:
            self.dedent()
            self.write("else")
            self.indent()
            self.write('call %s("derived type array not allocated")' % self.abort_func)
            self.dedent()
            self.write("end if")

        self.dedent()
        self.write("end subroutine %s" % (subroutine_name))
        self.write()

    def _write_array_len(self, t, el, sizeof_fortran_t):
        """
        Write a subroutine which returns the length of a derived-type array

        Parameters
        ----------
        t : `fortran.Type` node or `fortran.Module` node
            Node of the parse tree which contains this derived-type as an element

        el : `fortran.Element` node
            An element of a module which is derived-type array

        sizeof_fortan_t : `int`
            The size, in bytes, of a pointer to a fortran derived type ??
        """
        if isinstance(t, ft.Type):
            this = self.prefix + "this"
        else:
            # Module-level arrays do not need a this argument (issue #306).
            this = None
        safe_n = self.prefix + "n"  # YANN: "n" could be in the "uses"

        subroutine_name = "%s%s__array_len__%s" % (
            self.prefix,
            self._scope_identifier_for(t),
            el.name,
        )
        subroutine_name = shorten_long_name(subroutine_name)

        if this is not None:
            self.write("subroutine %s(%s, %s)" % (subroutine_name, this, safe_n))
        else:
            self.write("subroutine %s(%s)" % (subroutine_name, safe_n))
        self.indent()
        self.write()
        extra_uses = {}
        if isinstance(t, ft.Module):
            local_name = shorten_long_name("%s_%s" % (t.name, el.name))
            alias = "%s => %s" % (local_name, el.orig_name)
            self._add_extra_use(extra_uses, t.orig_name or t.name, alias)
        elif isinstance(t, ft.Type):
            if "super-type" in t.doc:
                # YANN: propagate parameter uses
                for use in t.uses:
                    module_name = use[0]
                    only_list = use[1] if len(use) > 1 else []
                    for symbol in only_list:
                        self._add_extra_use(extra_uses, module_name, symbol)
            else:
                self._add_extra_use(extra_uses, t.mod_name, None)

        owner_module = self._type_owner(el.type, getattr(t, "mod_name", getattr(t, "name", None)))
        self._add_extra_use(extra_uses, owner_module, None)
        self.write_uses_lines(el, extra_uses)
        # Add iso_c_binding for handle arrays (c_int is immune to -fdefault-integer-8)
        self.write("use, intrinsic :: iso_c_binding, only: c_int")
        self.write("implicit none")
        self.write()
        if "super-type" in t.doc:
            self.write_super_type_lines(t)

        # Check if the type has recursive definition:
        same_type = ft.strip_type(t.name) == ft.strip_type(el.type)
        if isinstance(t, ft.Type):
            self.write_type_or_class_lines(t.name)
        self.write_type_or_class_lines(el.type, same_type)
        self.write("integer(c_int), intent(out) :: %s" % (safe_n))
        if this is not None:
            self.write("integer(c_int), intent(in) :: %s(%d)" % (this, sizeof_fortran_t))
        if isinstance(t, ft.Type):
            self.write("type(%s_ptr_type) :: this_ptr" % t.name)
            self.write()
            self.write("this_ptr = transfer(%s, this_ptr)" % (this))
            array_name = self._get_type_member_array_name(t, el.name)
        else:
            array_name = shorten_long_name("%s_%s" % (t.name, el.name))

        if "allocatable" in el.attributes:
            self.write("if (allocated(%s)) then" % array_name)
            self.indent()

        self.write("%s = size(%s)" % (safe_n, array_name))

        if "allocatable" in el.attributes:
            self.dedent()
            self.write("else")
            self.indent()
            self.write("%s = 0" % (safe_n))
            self.dedent()
            self.write("end if")

        self.dedent()
        self.write("end subroutine %s" % (subroutine_name))
        self.write()

    def _write_scalar_wrapper(self, t, el, sizeof_fortran_t, getset):
        """
        Write get/set routines for scalar elements of derived-types and modules

        Parameters
        ----------
        t : `fortran.Type` node or `fortran.Module` node
            Node of the parse tree which contains this derived-type as an element

        el : `fortran.Element` node
            An element of a module which is derived-type array

        sizeof_fortan_t : `int`
            The size, in bytes, of a pointer to a fortran derived type ??

        getset : `str` {``"get"``,``"set"``}
            String indicating whether to write a get routine, or a set routine.
        """

        log.debug("writing %s wrapper for %s.%s" % (getset, t.name, el.name))

        # getset and inout just change things simply from a get to a set routine.
        inout = "in"
        if getset == "get":
            inout = "out"

        if isinstance(t, ft.Type):
            this = "this, "
        elif isinstance(t, ft.Module):
            this = ""
        else:
            raise ValueError(
                "Don't know how to write scalar wrappers for %s type %s" % (t, type(t))
            )

        # Get appropriate use statements
        extra_uses = {}
        module_local_name = None  # shortened name for module-level elements
        if isinstance(t, ft.Module):
            module_local_name = shorten_long_name("%s_%s" % (t.name, el.orig_name))
            alias = "%s => %s" % (module_local_name, el.orig_name)
            self._add_extra_use(extra_uses, t.orig_name or t.name, alias)
        elif isinstance(t, ft.Type):
            self._add_extra_use(extra_uses, t.mod_name, None)

        # Check if the type has recursive definition:
        same_type = ft.strip_type(t.name) == ft.strip_type(el.type)

        if el.type.startswith("type") and not same_type:
            owner_module = self._type_owner(el.type, getattr(t, "mod_name", getattr(t, "name", None)))
            self._add_extra_use(extra_uses, owner_module, None)

        # Prepend prefix to element name
        #   -- Since some cases require a safer localvar name, we always transform it
        localvar = self.prefix + el.orig_name

        subroutine_name = "%s%s__%s__%s" % (
            self.prefix,
            self._scope_identifier_for(t),
            getset,
            el.name,
        )
        subroutine_name = shorten_long_name(subroutine_name)

        self.write("subroutine %s(%s%s)" % (subroutine_name, this, localvar))
        self.indent()

        self.write_uses_lines(el, extra_uses)
        # Add iso_c_binding for handle arrays (c_int is immune to -fdefault-integer-8)
        if isinstance(t, ft.Type) or ft.is_derived_type(el.type):
            self.write("use, intrinsic :: iso_c_binding, only: c_int")

        self.write("implicit none")
        if isinstance(t, ft.Type):
            self.write_type_or_class_lines(t.orig_name)

        if ft.is_derived_type(el.type) and not (el.type == "type(" + t.name + ")"):
            self.write_type_or_class_lines(el.type, pointer=True)

        if isinstance(t, ft.Type):
            self.write("integer(c_int), intent(in)   :: this(%d)" % sizeof_fortran_t)
            self.write("type(%s_ptr_type) :: this_ptr" % t.orig_name)

        # Return/set by value
        attributes = [
            attr
            for attr in el.attributes
            if attr not in ["pointer", "allocatable", "public", "parameter", "save"]
        ]

        if ft.is_derived_type(el.type):
            # For derived types elements, treat as opaque reference
            self.write(
                "integer(c_int), intent(%s) :: %s(%d)" % (inout, localvar, sizeof_fortran_t)
            )

            self.write(
                "type(%s_ptr_type) :: %s_ptr" % (ft.strip_type(el.type), el.orig_name)
            )
            self.write()
            if isinstance(t, ft.Type):
                self.write("this_ptr = transfer(this, this_ptr)")
            if getset == "get":
                if isinstance(t, ft.Type):
                    if (self.is_class(el.type)):
                        self.write("allocate(%s_ptr%%p)" % el.orig_name)
                        source = "%s_ptr%%p%%obj =>" % el.orig_name
                    else:
                        source = "%s_ptr%%p =>" % el.orig_name
                    if (self.is_class(t.orig_name)):
                        target = "this_ptr%%p%%obj%%%s" % el.orig_name
                    else:
                        target = "this_ptr%%p%%%s" % el.orig_name
                    self.write("%s %s" % (source, target))
                else:
                    self.write(
                        "%s_ptr%%p => %s" % (el.orig_name, module_local_name)
                    )
                self.write(
                    "%s = transfer(%s_ptr,%s)" % (localvar, el.orig_name, localvar)
                )
            else:
                self.write(
                    "%s_ptr = transfer(%s,%s_ptr)"
                    % (el.orig_name, localvar, el.orig_name)
                )
                if isinstance(t, ft.Type):
                    if (self.is_class(el.type)):
                        target = "%s_ptr%%p%%obj" % el.orig_name
                    else:
                        target = "%s_ptr%%p" % el.orig_name
                    if (self.is_class(t.orig_name)):
                        source = "this_ptr%%p%%obj%%%s" % el.orig_name
                    else:
                        source = "this_ptr%%p%%%s" % el.orig_name
                    self.write("%s = %s" % (source, target))
                else:
                    self.write(
                        "%s = %s_ptr%%p" % (module_local_name, el.orig_name)
                    )
        else:
            if attributes != []:
                self.write(
                    "%s, %s, intent(%s) :: %s"
                    % (el.type, ",".join(attributes), inout, localvar)
                )
            else:
                self.write("%s, intent(%s) :: %s" % (el.type, inout, localvar))
            self.write()
            if isinstance(t, ft.Type):
                self.write("this_ptr = transfer(this, this_ptr)")
            if getset == "get":
                if isinstance(t, ft.Type):
                    if (self.is_class(t.orig_name)):
                        self.write("%s = this_ptr%%p%%obj%%%s" % (localvar, el.orig_name))
                    else:
                        self.write("%s = this_ptr%%p%%%s" % (localvar, el.orig_name))
                else:
                    self.write("%s = %s" % (localvar, module_local_name))
            else:
                if isinstance(t, ft.Type):
                    if (self.is_class(t.orig_name)):
                        self.write("this_ptr%%p%%obj%%%s = %s" % (el.orig_name, localvar))
                    else:
                        self.write("this_ptr%%p%%%s = %s" % (el.orig_name, localvar))
                else:
                    self.write("%s = %s" % (module_local_name, localvar))
        self.dedent()
        self.write("end subroutine %s" % (subroutine_name))
        self.write()
