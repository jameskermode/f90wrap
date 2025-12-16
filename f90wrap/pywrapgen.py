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

import os
import logging
import re
from typing import List
import numpy as np
from packaging import version

from f90wrap.transform import shorten_long_name, ArrayDimensionConverter
from f90wrap import fortran as ft
from f90wrap import codegen as cg

log = logging.getLogger(__name__)


def py_arg_value(arg):
    # made global from PythonWrapperGenerator.visit_Procedure so that other functions can use it
    if "optional" in arg.attributes or arg.value is None:
        return "=None"
    else:
        return ""


def normalise_class_name(name, name_map):
    return name_map.get(name.lower(), name.title())

class PythonWrapperGenerator(ft.FortranVisitor, cg.CodeGenerator):
    def __init__(
            self,
            prefix,
            mod_name,
            types,
            f90_mod_name=None,
            make_package=False,
            kind_map=None,
            init_file=None,
            py_mod_names=None,
            class_names=None,
            max_length=None,
            auto_raise=None,
            type_check=False,
            relative=False,
            return_decoded=False,
            return_bool=False,
            namespace_types=False):
        if max_length is None:
            max_length = 80
        cg.CodeGenerator.__init__(
            self, indent=" " * 4, max_length=max_length, continuation="\\", comment="#"
        )
        ft.FortranVisitor.__init__(self)
        self.prefix = prefix
        self.py_mod_name = mod_name
        self.py_mod_names = py_mod_names
        self.class_names = class_names
        if f90_mod_name is None:
            f90_mod_name = "_" + mod_name
        self.f90_mod_name = f90_mod_name
        self.types = types
        self.imports = set()
        self.make_package = make_package
        if kind_map is None:
            kind_map = {}
        self.kind_map = kind_map
        self.init_file = init_file
        self.type_check = type_check
        self.relative = relative
        self.return_decoded = return_decoded
        self.return_bool = return_bool
        try:
            self._err_num_var, self._err_msg_var = auto_raise.split(',')
        except ValueError:
            self._err_num_var, self._err_msg_var = None, None

        if version.parse(np.version.version) < version.parse("2.0"):
            self.numpy_complexwarning = "numpy.ComplexWarning"
        else:
            self.numpy_complexwarning = "numpy.exceptions.ComplexWarning"
        self._namespace_types = namespace_types

    def _scope_identifier_for(self, container):
        """Return stable identifier used for generated helper names."""
        if isinstance(container, ft.Module) or not self._namespace_types:
            return container.name
        if isinstance(container, ft.Type):
            owner = getattr(container, "mod_name", None)
            if owner is None:
                type_key = ft.strip_type(container.name)
                type_node = self.types.get(type_key)
                owner = getattr(type_node, "mod_name", None) if type_node is not None else None
            if owner:
                return f"{owner}__{container.name}"
            return container.name
        raise TypeError("Unsupported scope container %r" % (container,))

    def _destructor_proc_for(self, constructor):
        """Return the destructor procedure associated with a constructor."""
        type_name = getattr(constructor, "type_name", None)
        if not type_name:
            return None

        type_key = ft.strip_type(type_name)
        type_node = self.types.get(type_key)
        if type_node is None:
            return None

        def _has_destructor(entity):
            attributes = getattr(entity, "attributes", []) or []
            return any(attr.strip().lower() == "destructor" for attr in attributes)

        for proc in getattr(type_node, "procedures", []) or []:
            if _has_destructor(proc):
                return proc

        for interface in getattr(type_node, "interfaces", []) or []:
            if not _has_destructor(interface):
                continue
            for proc in getattr(interface, "procedures", []) or []:
                if _has_destructor(proc):
                    return proc

        for binding in getattr(type_node, "bindings", []) or []:
            if not _has_destructor(binding):
                continue
            for proc in getattr(binding, "procedures", []) or []:
                if _has_destructor(proc):
                    return proc

        return None

    @staticmethod
    def _helper_name_for_proc(prefix, proc):
        """Return the generated helper name for a procedure."""
        helper_name = proc.name
        if getattr(proc, "mod_name", None):
            helper_name = f"{proc.mod_name}__{helper_name}"
        return shorten_long_name(f"{prefix}{helper_name}")

    def _finalizer_arguments(self, proc):
        """Return positional argument expressions for destructor helper.

        We currently support only the common case of a single derived-type
        argument (e.g. `this`).  For other signatures we fall back to the
        legacy behaviour.
        """
        arguments = getattr(proc, "arguments", []) or []
        if len(arguments) != 1:
            return None

        arg = arguments[0]
        arg_type = getattr(arg, "type", "")
        if isinstance(arg_type, str) and arg_type.lower().startswith(("type(", "class(")):
            return ["self._handle"]
        if arg.name == "this":
            return ["self._handle"]

        return None

    def write_imports(self, insert=0):
        default_imports = [
            (self.f90_mod_name, None),
            ("f90wrap.runtime", None),
            ("logging", None),
            ("numpy", None),
            ("warnings", None),
            ("weakref", None),
        ]
        if self.relative: default_imports[0] = ('..', self.f90_mod_name)
        imp_lines = ['from __future__ import print_function, absolute_import, division']
        for (mod, symbol) in default_imports + list(self.imports):
            if symbol is None:
                symbol_str = mod.partition('.')[2]
                if self.relative and mod.startswith(self.py_mod_name):
                    imp_lines.append("from . import %s" % (symbol_str))
                else:
                    imp_lines.append('import %s' % mod)
            else:
                submodule = mod.partition('.')[2]
                if isinstance(symbol, tuple):
                    symbol_str = ", ".join(symbol)
                else:
                    symbol_str = symbol

                if self.relative and mod.startswith(self.py_mod_name):
                    imp_lines.append("from .%s import %s" % (submodule, symbol_str))
                else:
                    imp_lines.append("from %s import %s" % (mod, symbol_str))


        imp_lines += ["\n"]
        self.imports = set()
        return self.writelines(imp_lines, insert=insert, level=0)

    def visit_Root(self, node):
        """
        Wrap subroutines and functions that are outside of any Fortran modules
        """
        if self.make_package:
            if not os.path.exists(self.py_mod_name):
                os.mkdir(self.py_mod_name)

        self.code = []
        self.py_mods = []
        self.current_module = None

        self.generic_visit(node)

        if self.make_package:
            for py_mod in self.py_mods:
                self.imports.add(
                    (
                        self.py_mod_name + "." + self.py_mod_names.get(py_mod, py_mod),
                        None,
                    )
                )
        self.write_imports(0)

        if self.make_package:
            py_wrapper_file = open(os.path.join(self.py_mod_name, "__init__.py"), "w")
        else:
            py_wrapper_file = open("%s.py" % self.py_mod_name, "w")
        py_wrapper_file.write(str(self))
        if self.init_file is not None:
            py_wrapper_file.write(open(self.init_file).read())
        py_wrapper_file.close()

    def visit_Module(self, node):
        log.info("PythonWrapperGenerator visiting module %s %s" % (node.name, type(node)))
        if node.is_external:
            log.info("PythonWrapperGenerator skip external module %s" % node.name)
            self.current_module = None
            return
        cls_name = normalise_class_name(node.name, self.class_names)
        node.array_initialisers = []
        node.dt_array_initialisers = []
        self.current_module = self.py_mod_names.get(node.name, node.name)

        if self.make_package:
            self.code = []
            self.write(self._format_doc_string(node))
        else:
            self.write("class %s(f90wrap.runtime.FortranModule):" % cls_name)
            self.indent()
            self.write(self._format_doc_string(node))

            if (
                len(node.elements) == 0
                and len(node.types) == 0
                and len(node.procedures) == 0
            ):
                self.write("pass")

        index = len(self.code)  # save position to insert import lines

        self.generic_visit(node)

        properties = []  # Collect list of properties for a __repr__()
        for el in node.elements:
            dims = list(filter(lambda x: x.startswith("dimension"), el.attributes))
            if len(dims) == 0:  # proper scalar type (normal or derived)
                if el.type.startswith(("type", "class")):
                    self.write_dt_wrappers(node, el, properties)
                else:
                    self.write_scalar_wrappers(node, el, properties)
            elif el.type.startswith(("type", "class")):  # array of derived types
                self.write_dt_array_wrapper(node, el, dims[0])
            else:
                self.write_sc_array_wrapper(node, el, dims[0], properties)
        self.write_repr(node, properties)

        # insert import statements at the beginning of module
        if self.make_package:
            index = self.write_imports(index)
            index = self.writelines(['logger = logging.getLogger(__name__)'], insert=index)
            index = self.writelines(
                [f'warnings.filterwarnings("error", category={self.numpy_complexwarning})'],
                insert=index
            )
            self.writelines(["_arrays = {}", "_objs = {}", "\n"], insert=index)
            self.write()

        if self.make_package:
            self.write(
                "_array_initialisers = [%s]" % (", ".join(node.array_initialisers))
            )
        self.write(
            "_dt_array_initialisers = [%s]" % (", ".join(node.dt_array_initialisers))
        )
        self.write()

        proc_lookup = {proc.name: self.prefix + (f"{proc.mod_name}__" if proc.mod_name else "") + proc.name
                       for proc in getattr(node, 'procedures', [])}
        proc_lookup = {name: shorten_long_name(helper) for name, helper in proc_lookup.items()}

        module_proc_names = {proc.name for proc in getattr(node, "procedures", [])}
        fallback_bindings: List[str] = []

        for derived in getattr(node, "types", []):
            for binding in getattr(derived, "bindings", []):
                if getattr(binding, "type", None) != "procedure":
                    continue
                targets = getattr(binding, "procedures", [])
                if not targets:
                    continue
                target_name = getattr(targets[0], "name", None)
                helper_name = proc_lookup.get(target_name)
                candidates = []
                if helper_name:
                    candidates.append(helper_name)
                else:
                    # Try plain module prefix + target name as a fallback
                    candidates.append(
                        shorten_long_name(
                            f"{self.prefix}{node.name}__{target_name}"
                        )
                    )
                alias = shorten_long_name(
                    f"{self.prefix}{node.name}__{binding.name}__binding__{derived.name.lower()}"
                )
                if candidates:
                    candidate_list = ", ".join(f'"{name}"' for name in candidates)
                    self.write(
                        f"if not hasattr({self.f90_mod_name}, \"{alias}\"):")
                    self.indent()
                    self.write(
                        f"for _candidate in [{candidate_list}]:"
                    )
                    self.indent()
                    self.write(
                        f"if hasattr({self.f90_mod_name}, _candidate):")
                    self.indent()
                    self.write(
                        f"setattr({self.f90_mod_name}, \"{alias}\", getattr({self.f90_mod_name}, _candidate))")
                    self.write("break")
                    self.dedent()
                    self.dedent()
                    self.dedent()
                if (
                    binding.name not in module_proc_names
                    and binding.name.isidentifier()
                    and not binding.name.startswith("p_")
                ):
                    fallback_bindings.append(binding.name)
        if getattr(node, "types", []):
            self.write()

        for binding_name in sorted(set(fallback_bindings)):
            self.write("@staticmethod")
            self.write(f"def {binding_name}(instance, *args, **kwargs):")
            self.indent()
            self.write(f"return instance.{binding_name}(*args, **kwargs)")
            self.dedent()
            self.write()

        # FIXME - make this less ugly, e.g. by generating code for each array
        if self.make_package:
            self.write(
                """try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module "%s".')
"""
                % node.name
            )
            self.write()
            self.write(
                """for func in _dt_array_initialisers:
    func()
            """
            )
            if len(self.code) > 0:
                py_mod_name = self.py_mod_names.get(node.name, node.name)
                py_wrapper_file = open(
                    os.path.join(self.py_mod_name, py_mod_name + ".py"), "w"
                )
                py_wrapper_file.write(str(self))
                py_wrapper_file.close()
                self.py_mods.append(node.name)
            self.code = []
        else:
            self.dedent()  # finish the FortranModule class
            self.write()
            # instantiate the module class using mapped name (issue #269)
            self.write("%s = %s()" % (self.current_module, cls_name))
            self.write()

        self.current_module = None

    def write_constructor(self, node):
        if "abstract" in node.attributes:
            self.write("def __init__(self):")
            self.indent()
            self.write('raise(NotImplementedError("This is an abstract class"))')
            self.dedent()
            self.write()
            # Abstract classes still need _setup_finalizer for polymorphic factory returns
            self.write("def _setup_finalizer(self):")
            self.indent()
            self.write('"""Abstract classes have no destructor to call."""')
            self.write("pass")
            self.dedent()
            self.write()
            return

        handle_arg = ft.Argument(
            name="handle",
            filename=node.filename,
            doc=["Opaque reference to existing derived type instance"],
            lineno=node.lineno,
            attributes=["intent(in)", "optional"],
            type="integer",
        )
        handle_arg.py_name = "handle"

        # special case for constructors: return value is 'self' argument,
        # plus we add an extra optional argument
        args = node.arguments + [handle_arg]

        dct = dict(
            func_name=node.name,
            prefix=self.prefix,
            mod_name=self.f90_mod_name,
            py_arg_names=", ".join(
                [
                    "%s%s"
                    % (arg.py_name, "optional" in arg.attributes and "=None" or "")
                    for arg in args
                ]
            ),
            f90_arg_names=", ".join(
                ["%s=%s" % (arg.name, arg.py_value) for arg in node.arguments]
            ),
        )

        if node.mod_name is not None:
            dct["func_name"] = node.mod_name + "__" + node.name
        dct["subroutine_name"] = shorten_long_name("%(prefix)s%(func_name)s" % dct)

        self.write("def __init__(self, %(py_arg_names)s):" % dct)
        self.indent()
        self.write(self._format_doc_string(node))
        for arg in node.arguments:
            if "optional" in arg.attributes and "._handle" in arg.py_value:
                dct["f90_arg_names"] = dct["f90_arg_names"].replace(
                    arg.py_value,
                    (
                        "(None if %(arg_py_name)s is None else %("
                        "arg_py_name)s._handle)"
                    )
                    % {"arg_py_name": arg.py_name},
                )
        self.write("f90wrap.runtime.FortranDerivedType.__init__(self)")

        self.write("if isinstance(handle, numpy.ndarray) and handle.ndim == 1 and handle.dtype.num == 5:")
        self.indent()
        self.write("self._handle = handle")
        self.write("self._alloc = True")
        self.dedent()
        self.write("else:")
        self.indent()
        self.write(
            "result = %(mod_name)s.%(subroutine_name)s(%(f90_arg_names)s)" % dct
        )
        self.write(
            "self._handle = result[0] if isinstance(result, tuple) else result"
        )
        self.write("self._alloc = True")
        self.dedent()

        # Call the _setup_finalizer method to register the destructor
        self.write("self._setup_finalizer()")
        self.dedent()
        self.write()

        # Generate the _setup_finalizer method that can be called from __init__ or from_handle
        destructor_proc = self._destructor_proc_for(node)
        fallback_call = "%(mod_name)s.%(subroutine_name)s" % dct
        # Replace _initialise with _finalise for fallback finalizer
        # Note: use single underscore pattern, not double, because subroutine names
        # like "f90wrap_module__type_initialise" have __ between module and type,
        # but only _ between type and "initialise"
        fallback_call = fallback_call.replace("_initialise", "_finalise")
        destructor_helper = None
        destructor_args = None

        if destructor_proc is not None:
            destructor_helper = self._helper_name_for_proc(self.prefix, destructor_proc)
            destructor_args = self._finalizer_arguments(destructor_proc)
            if destructor_args is None:
                log.debug(
                    "Unsupported destructor signature for %s; using legacy finalizer",
                    getattr(node, "type_name", "<unknown>"),
                )
                destructor_helper = None
        else:
            log.debug(
                "No destructor found for type %s; falling back to legacy finalizer",
                getattr(node, "type_name", "<unknown>"),
            )

        self.write("def _setup_finalizer(self):")
        self.indent()
        self.write('"""Set up weak reference destructor to prevent Fortran memory leaks."""')
        # Register finalizer using weakref (modern Python 3.4+ approach)
        # More reliable than __del__ for resource cleanup
        self.write("if self._alloc:")
        self.indent()
        if destructor_helper and destructor_args is not None:
            self.write(f"destructor = getattr({self.f90_mod_name}, \"{destructor_helper}\")")
            if destructor_args:
                args_literal = ", ".join(destructor_args)
                self.write(f"self._finalizer = weakref.finalize(self, destructor, {args_literal})")
            else:
                self.write("self._finalizer = weakref.finalize(self, destructor)")
        else:
            self.write(f"self._finalizer = weakref.finalize(self, {fallback_call}, self._handle)")
        self.dedent()
        self.dedent()
        self.write()

    def write_classmethod(self, node):
        dct = dict(
            func_name=node.name,
            method_name=hasattr(node, "method_name") and node.method_name or node.name,
            prefix=self.prefix,
            mod_name=self.f90_mod_name,
            py_arg_names=", ".join(
                [arg.py_name + py_arg_value(arg) for arg in node.arguments]
            ),
            f90_arg_names=", ".join(
                ["%s=%s" % (arg.name, arg.py_value) for arg in node.arguments]
            ),
            call="",
        )

        dct["call"] = "result = "
        for arg in node.arguments:
            if 'optional' in arg.attributes and '._handle' in arg.py_value:
                dct['f90_arg_names'] = dct['f90_arg_names'].replace(arg.py_value,
                                                                    ('None if %(arg_py_name)s is None else %('
                                                                     'arg_py_name)s._handle') %
                                                                    {'arg_py_name': arg.py_name})
        if node.mod_name is not None:
            dct['func_name'] = node.mod_name + '__' + node.name
        dct['subroutine_name'] = shorten_long_name('%(prefix)s%(func_name)s' % dct)
        call_line = '%(call)s%(mod_name)s.%(subroutine_name)s(%(f90_arg_names)s)' % dct

        self.write("@classmethod")
        self.write("def %(method_name)s(cls, %(py_arg_names)s):" % dct)
        self.indent()
        self.write(self._format_doc_string(node))
        self.write("bare_class = cls.__new__(cls)")
        self.write("f90wrap.runtime.FortranDerivedType.__init__(bare_class)")

        self.write(call_line)

        self.write(
            "bare_class._handle = result[0] if isinstance(result, tuple) else result"
        )
        self.write("bare_class._alloc = True")
        self.write("return bare_class")

        self.dedent()
        self.write()

    def write_destructor(self, node):
        # DEPRECATED: __del__ method generation disabled in favor of weakref.finalize
        # weakref.finalize is more reliable and deterministic for resource cleanup
        # The finalizer is now registered in write_constructor()
        # This method is kept for backward compatibility but does not generate code
        pass

    def visit_Procedure(self, node):
        log.info("PythonWrapperGenerator visiting routine %s" % node.name)

        self._filtered_arguments = node.arguments
        if isinstance(node, ft.Function):
            self._filtered_ret_val = node.ret_val

        if "constructor" in node.attributes:
            self.write_constructor(node)
        elif "destructor" in node.attributes:
            self.write_destructor(node)
        elif "classmethod" in node.attributes:
            self.write_classmethod(node)
        elif "abstract" in node.attributes and not "method" in node.attributes:
            return self.generic_visit(node)

        else:
            dct = dict(
                func_name=node.name,
                method_name=(hasattr(node, "binding_name") and node.binding_name)
                or (hasattr(node, "method_name") and node.method_name)
                or node.name,
                prefix=self.prefix,
                mod_name=self.f90_mod_name,
                py_arg_names=", ".join(
                    [arg.py_name + py_arg_value(arg) for arg in node.arguments]
                ),
                f90_arg_names=", ".join(
                    ["%s=%s" % (arg.name, arg.py_value) for arg in node.arguments]
                ),
                call="",
            )
            if node.mod_name is not None:
                dct['func_name'] = node.mod_name + '__' + node.name
            dct['subroutine_name'] = shorten_long_name('%(prefix)s%(func_name)s' % dct)

            if self._err_num_var is not None and self._err_msg_var is not None:
                self._filtered_arguments = [
                    arg
                    for arg in self._filtered_arguments
                    if arg.name not in [self._err_num_var, self._err_msg_var]
                ]
                if isinstance(node, ft.Function):
                    self._filtered_ret_val = [
                        ret_val
                        for ret_val in self._filtered_ret_val
                        if ret_val.name not in [self._err_num_var, self._err_msg_var]
                    ]

            if isinstance(node, ft.Function):
                dct["result"] = ", ".join(
                    [ret_val.name for ret_val in self._filtered_ret_val]
                )
                dct["call"] = ", ".join([ret_val.name for ret_val in self._filtered_ret_val])
                if dct["call"]:
                    dct["call"] = dct["call"] + " = "

            py_sign_names = [
                arg.py_name + py_arg_value(arg) for arg in self._filtered_arguments
            ]
            f90_call_names = [
                "%s=%s" % (arg.name, arg.py_value) if arg.py_value else "%s" % arg.name
                for arg in self._filtered_arguments
            ]

            # Add optional argument to specify if function is called from interface
            py_sign_names.append("interface_call=False")

            dct["py_arg_names"] = ", ".join(py_sign_names)
            dct["f90_arg_names"] = ", ".join(f90_call_names)

            if (
                not self.make_package
                and node.mod_name is not None
                and node.type_name is None
            ):
                # procedures outside of derived types become static methods
                self.write("@staticmethod")

            self.write("def %(method_name)s(%(py_arg_names)s):" % dct)
            self.indent()
            self.write(self._format_doc_string(node))

            if self.type_check:
                self.write_type_checks(node)

            for arg in self._filtered_arguments:
                if "optional" in arg.attributes and "._handle" in arg.py_value:
                    dct["f90_arg_names"] = dct["f90_arg_names"].replace(
                        arg.py_value,
                        ("None if %(arg_py_name)s is None else %(arg_py_name)s._handle")
                        % {"arg_py_name": arg.py_name},
                    )
            # Add dimension argument for fortran functions that returns an array
            if isinstance(node, ft.Function):

                def f902py_name(node, f90_name):
                    for arg in self._filtered_arguments:
                        if arg.name == f90_name:
                            return arg.py_name
                    return ""

                args_py_names = [arg.py_name for arg in self._filtered_arguments]
                offset = 0
                # Regular arguments are first, compute the index offset
                for arg in self._filtered_arguments:
                    dynamic_dims = [d for d in arg.dims_list() if not ArrayDimensionConverter.valid_dim_re.match(d.strip())]
                    offset += len(dynamic_dims)
                for retval in self._filtered_ret_val:
                    dynamic_dims = [d for d in retval.dims_list() if not ArrayDimensionConverter.valid_dim_re.match(d.strip())]
                    for dim_str in dynamic_dims:
                        # remove unnecessary '1:' prefix, e.g. 1:n or 1:size(x)
                        if dim_str.startswith("1:"):
                            dim_str = dim_str[2:]
                        elif ":" in dim_str:
                            log.error("Cannot wrap ranges for dimension arguments: %s" % dim_str)

                        # Both "size" and "len" are replaced by "size_bn" and "len_bn"
                        # ("badname") by numpy.f2py. Try both patterns.
                        match = None
                        for keyword in ["size", "len"]:
                            try:
                                keyword_bn = np.f2py.crackfortran.badnames[keyword]
                            except KeyError:
                                keyword_bn = keyword
                            match = re.search(r"%s\((.*)\)" % keyword_bn, dim_str)
                            if match:
                                break

                        if match:
                            # Case where return size is size/len of input
                            size_arg = match.group(1).split(",")
                            py_name = f902py_name(node, size_arg[0])
                            try:
                                dim_num = int(size_arg[1]) - 1
                            except IndexError:
                                dim_num = 0
                            out_dim = "%s.shape[%d]" % (py_name, dim_num)
                        else:
                            # Case where return size is input
                            py_name = f902py_name(node, dim_str.split("%")[0])
                            # It could be a member of an object
                            members_arg = dim_str.split("%")[1:]
                            if members_arg:
                                out_dim = "%s.%s" % (py_name, ".".join(members_arg))
                            else:
                                out_dim = "%s" % (py_name)

                        if py_name in args_py_names:
                            log.info("Adding dimension argument to '%s' ('%s' -> '%s')" % (node.name, dim_str, out_dim))
                            dct["f90_arg_names"] = "%s, %s" % (
                                dct["f90_arg_names"],
                                "f90wrap_n%d=%s" % (offset, out_dim),
                            )
                        else:
                            log.error("Failed adding dimension argument to '%s' ('%s' -> '%s')" % (node.name, dim_str, out_dim))
                        offset += 1

            call_line = (
                "%(call)s%(mod_name)s.%(subroutine_name)s(%(f90_arg_names)s)" % dct
            )
            self.write(call_line)

            if isinstance(node, ft.Function):
                # convert any derived type return values to Python objects
                for ret_val in self._filtered_ret_val:
                    if ret_val.type.startswith(("type", "class")):
                        cls_name = normalise_class_name(
                            ft.strip_type(ret_val.type), self.class_names
                        )
                        py_mod_name = self.py_mod_name
                        if hasattr(self.types[ft.strip_type(ret_val.type)], "py_mod_name"):
                            py_mod_name = self.types[ft.strip_type(ret_val.type)].py_mod_name
                        cls_name = py_mod_name + "." + cls_name
                        cls_name = 'f90wrap.runtime.lookup_class("%s")' % cls_name
                        cls_mod_name = self.types[ft.strip_type(ret_val.type)].mod_name
                        cls_mod_name = self.py_mod_names.get(cls_mod_name, cls_mod_name)
                        # if self.make_package:
                        #     if cls_mod_name != self.current_module:
                        #         self.imports.add((self.py_mod_name + '.' + cls_mod_name, cls_name))
                        # else:
                        #     cls_name = cls_mod_name + '.' + cls_name
                        self.write(
                            "%s = %s.from_handle(%s, alloc=True)"
                            % (ret_val.name, cls_name, ret_val.name)
                        )
                        # Set up finalizer for objects created via factory functions
                        self.write("%s._setup_finalizer()" % ret_val.name)
                    # strip white space for string returns
                    pytype = ft.f2py_type(ret_val.type)
                    if self.return_decoded and pytype == "str":
                        dct["result"] = dct["result"].replace(
                            ret_val.name, '%s.strip().decode("utf-8")' % ret_val.name
                        )
                    # convert back Fortran logical to Python bool
                    if self.return_bool and ret_val.type == "logical":
                        dct["result"] = dct["result"].replace(
                            ret_val.name, 'bool(%s)' % ret_val.name
                        )

                if dct["result"]:
                    self.write("return %(result)s" % dct)

            self.dedent()
            self.write()

    def _sort_procedures_by_specificity(self, procedures):
        """
        Sort procedures by specificity to improve overload resolution.

        More specific procedures (array parameters) should be tried before
        less specific ones (scalar parameters) because f2py silently accepts
        arrays for scalar parameters by taking the first element, which
        prevents fallback to the correct array version.

        Returns procedures sorted with array versions first, scalar versions last.
        """
        def _count_array_params(proc):
            """Count number of array parameters in procedure signature"""
            array_count = 0
            for arg in proc.arguments:
                # Check if argument has dimension attribute
                if any(attr.startswith('dimension') for attr in arg.attributes):
                    array_count += 1
            return array_count

        def specificity_key(proc):
            """
            Return a sort key where higher values = more specific = try first.

            Procedures with array parameters get higher scores than scalar-only.
            Among procedures with same array count, more total parameters = more specific.
            """
            if isinstance(proc, ft.Prototype):
                return 0  # Prototypes are least specific
            array_params = _count_array_params(proc)
            total_params = len(proc.arguments)

            # Array parameters weighted heavily (x100) to ensure they come first
            return (array_params * 100) + total_params

        # Sort in descending order (most specific first)
        return sorted(procedures, key=specificity_key, reverse=True)

    def write_exception_handler(self, dct):
        # try to call each in turn until no TypeError raised
        self.write("for proc in %(proc_names)s:" % dct)
        self.indent()
        self.write("exception=None")
        self.write("try:")
        self.indent()
        self.write("return proc(*args, **kwargs, interface_call=True)")
        self.dedent()
        self.write(
            f"except (TypeError, ValueError, AttributeError, IndexError, {self.numpy_complexwarning}) as err:"
        )
        self.indent()
        self.write("exception = \"'%s: %s'\" % (type(err).__name__, str(err))")
        self.write("continue")
        self.dedent()
        self.dedent()
        self.write()

        self.write("argTypes=[]")
        self.write("for arg in args:")
        self.indent()
        self.write("try:")
        self.indent()
        self.write(
            "argTypes.append(\"%s: dims '%s', type '%s',\"\n\" type code '%s'\"\n%(str(type(arg)),"
            "arg.ndim, arg.dtype, arg.dtype.num))"
        )
        self.dedent()
        self.write("except AttributeError:")
        self.indent()
        self.write("argTypes.append(str(type(arg)))")
        self.dedent()
        self.dedent()

        self.write('raise TypeError("Not able to call a version of "')
        self.indent()
        self.write('"%(intf_name)s compatible with the provided args:"' % dct)
        self.write(
            '"\\n%s\\nLast exception was: %s"%("\\n".join(argTypes), exception))'
        )
        self.dedent()
        self.dedent()
        self.write()

    def visit_Binding(self, node):
        # Handle generic binding similary as interfaces
        # Leave other binding type alone
        if node.type != "generic":
            return self.generic_visit(node)

        log.info("PythonWrapperGenerator visiting generic binding %s" % node.name)

        # first output all the procedures within the interface
        self.generic_visit(node)

        proc_names = []
        for proc in node.procedures:
            proc_name = "self.%s" % proc.name
            proc_names.append(proc_name)

        dct = dict(
            intf_name=node.method_name, proc_names="[" + ", ".join(proc_names) + "]"
        )
        self.write("def %(intf_name)s(self, *args, **kwargs):" % dct)
        self.indent()
        self.write(self._format_doc_string(node))
        self.write_exception_handler(dct)

    def visit_Interface(self, node):
        log.info("PythonWrapperGenerator visiting interface %s" % node.name)

        if "abstract" in node.attributes:
            log.info(" -> abstract interface, skipping")
            return

        # first output all the procedures within the interface
        self.generic_visit(node)
        cls_name = None
        if node.type_name is not None:
            cls_name = normalise_class_name(
                ft.strip_type(node.type_name), self.class_names
            )
        # Check if any procedure has the same name as the interface (will be shadowed)
        shadowed_methods = set()
        log.info(f"PythonWrapperGenerator: Interface {node.name} has {len(node.procedures)} procedures")
        for proc in node.procedures:
            method_name = proc.method_name if hasattr(proc, "method_name") else proc.name
            log.info(f"  Procedure: {proc.name}, method_name: {method_name}, interface method_name: {node.method_name}")
            if method_name == node.method_name:
                log.info(f"    -> Will be shadowed!")
                shadowed_methods.add(method_name)

        # Sort procedures by specificity (array versions before scalar)
        sorted_procedures = self._sort_procedures_by_specificity(node.procedures)
        proc_names = []
        for i, proc in enumerate(sorted_procedures):
            proc_name = ""
            if not self.make_package and hasattr(proc, "mod_name"):
                proc_name += normalise_class_name(proc.mod_name, self.class_names) + "."
            elif cls_name is not None:
                proc_name += cls_name + "."

            method_name = proc.method_name if hasattr(proc, "method_name") else proc.name

            # Use saved reference name if this method will be shadowed
            if method_name in shadowed_methods:
                proc_name += f"_{method_name}_{i}"
            else:
                proc_name += method_name
            proc_names.append(proc_name)

        # Write code to save references before the overloaded method is defined
        if shadowed_methods:
            self.write()
            self.write("# Save references to the original methods before overloading")
            for i, proc in enumerate(sorted_procedures):
                method_name = proc.method_name if hasattr(proc, "method_name") else proc.name
                if method_name in shadowed_methods:
                    self.write(f"_{method_name}_{i} = {method_name}")
            self.write()

        dct = dict(
            intf_name=node.method_name, proc_names="[" + ", ".join(proc_names) + "]"
        )
        if not self.make_package:
            # procedures outside of derived types become static methods
            self.write("@staticmethod")
        self.write("def %(intf_name)s(*args, **kwargs):" % dct)
        self.indent()
        self.write(self._format_doc_string(node))
        self.write_exception_handler(dct)

    def visit_Type(self, node):
        log.info("PythonWrapperGenerator visiting type %s" % node.name)
        node.dt_array_initialisers = []
        cls_name = normalise_class_name(node.name, self.class_names)
        cls_parent = "f90wrap.runtime.FortranDerivedType"
        if node.parent:
            cls_parent = normalise_class_name(node.parent.name, self.class_names)
            if node.parent.mod_name != node.mod_name:
                cls_parent = "%s.%s" % (node.parent.mod_name, cls_parent)
                if self.make_package:
                    if self.relative:
                        py_mod_name = '.'
                    else:
                        py_mod_name = self.py_mod_name
                    if hasattr(node.parent, "py_mod_name"):
                        py_mod_name = node.parent.py_mod_name
                    self.imports.add((py_mod_name, node.parent.mod_name))
        self.write(
            '@f90wrap.runtime.register_class("%s.%s")' % (self.py_mod_name, cls_name)
        )
        self.write("class %s(%s):" % (cls_name, cls_parent))
        self.indent()
        self.write(self._format_doc_string(node))
        self.generic_visit(node)

        self.write_member_variables(node)

        self.write()
        self.dedent()
        self.write()

    def write_member_variables(self, node):
        properties = []
        for el in node.elements:
            dims = list(filter(lambda x: x.startswith("dimension"), el.attributes))
            if len(dims) == 0:  # proper scalar type (normal or derived)
                if el.type.startswith(("type", "class")):
                    self.write_dt_wrappers(node, el, properties)
                else:
                    self.write_scalar_wrappers(node, el, properties)
            elif el.type.startswith(("type", "class")):  # array of derived types
                self.write_dt_array_wrapper(node, el, dims[0])
            else:
                self.write_sc_array_wrapper(node, el, dims, properties)
        self.write_repr(node, properties)

        self.write(
            "_dt_array_initialisers = [%s]" % (", ".join(node.dt_array_initialisers))
        )

    def write_scalar_wrappers(self, node, el, properties):
        dct = dict(
            el_name=el.name,
            el_orig_name=el.orig_name,
            el_name_get=el.name,
            el_name_set=el.name,
            mod_name=self.f90_mod_name,
            prefix=self.prefix,
            type_name=node.name,
            scope_name=self._scope_identifier_for(node),
            self="self",
            selfdot="self.",
            selfcomma="self, ",
            handle=isinstance(node, ft.Type) and "self._handle" or "",
        )

        if hasattr(el, "py_name"):
            dct["el_name_get"] = el.py_name
            dct["el_name_set"] = el.py_name

        if isinstance(node, ft.Type):
            dct["set_args"] = "%(handle)s, %(el_name_get)s" % dct
        else:
            dct["set_args"] = "%(el_name_get)s" % dct

        if not isinstance(node, ft.Module) or not self.make_package:
            self.write("@property")
            properties.append(el)
        else:
            dct["el_name_get"] = "get_" + el.name
            dct["el_name_set"] = "set_" + el.name
            dct["self"] = ""
            dct["selfdot"] = ""
            dct["selfcomma"] = ""

        # check for name clashes with pre-existing routines
        procedure_names = []
        if hasattr(node, "procedures"):
            procs = [proc.name for proc in node.procedures]
            procedure_names = procs
            if dct["el_name_get"] in procs:
                dct["el_name_get"] += "_"
            if dct["el_name_set"] in procs:
                dct["el_name_set"] += "_"

        dct['subroutine_name_get'] = shorten_long_name('%(prefix)s%(scope_name)s__get__%(el_name)s' % dct)
        dct['subroutine_name_set'] = shorten_long_name('%(prefix)s%(scope_name)s__set__%(el_name)s' % dct)

        self.write("def %(el_name_get)s(%(self)s):" % dct)
        self.indent()
        self.write(self._format_doc_string(el))
        self.write('return %(mod_name)s.%(subroutine_name_get)s(%(handle)s)' % dct)
        self.dedent()
        self.write()
        if (
            "parameter" in el.attributes
            and isinstance(node, ft.Module)
            and self.make_package
        ):
            self.write("%(el_orig_name)s = %(el_name_get)s()" % dct)
            self.write()

        if "parameter" not in el.attributes:
            if not isinstance(node, ft.Module) or not self.make_package:
                self.write('@%(el_name_get)s.setter' % dct)
            self.write('''def %(el_name_set)s(%(selfcomma)s%(el_name)s):
    %(mod_name)s.%(subroutine_name_set)s(%(set_args)s)
    ''' % dct)
            self.write()

        if isinstance(node, ft.Module) and dct.get("selfdot"):
            getter_name = "get_%s" % el.name
            self.write(f"def {getter_name}(self):")
            self.indent()
            self.write(f"return self.{el.name}")
            self.dedent()
            self.write()
            if "parameter" not in el.attributes:
                forward_name = "set_%s" % el.name
                if forward_name in procedure_names:
                    forward_name += "_value"
                self.write(f"def {forward_name}(self, value):")
                self.indent()
                self.write(f"self.{el.name} = value")
                self.dedent()
                self.write()

    def write_repr(self, node, properties):
        if len(properties) < 1:
            return

        self.write("def __str__(self):")

        self.indent()
        self.write(r"ret = ['<{0}>".format(node.name) + r"{\n']")
        self.write("ret.append('    {0} : ')".format(properties[0].name))
        self.write("ret.append(repr(self.{0}))".format(properties[0].name))
        for el in properties[1:]:
            self.write(r"ret.append(',\n    {0} : ')".format(el.name))
            self.write("ret.append(repr(self.{0}))".format(el.name))
        self.write("ret.append('}')")
        self.write("return ''.join(ret)")
        self.dedent()
        self.write()

    def write_dt_wrappers(self, node, el, properties):
        cls_name = normalise_class_name(ft.strip_type(el.type), self.class_names)
        mod_name = self.types[ft.strip_type(el.type)].mod_name
        cls_mod_name = self.py_mod_names.get(mod_name, mod_name)
        dct = dict(
            el_name=el.name,
            el_name_get=el.name,
            el_name_set=el.name,
            mod_name=self.f90_mod_name,
            prefix=self.prefix,
            type_name=node.name,
            scope_name=self._scope_identifier_for(node),
            cls_name=cls_name,
            cls_mod_name=cls_mod_name + ".",
            self="self",
            selfdot="self.",
            selfcomma="self, ",
            handle=isinstance(node, ft.Type) and "self._handle" or "",
        )
        if isinstance(node, ft.Type):
            dct["set_args"] = "%(handle)s, %(el_name)s" % dct
        else:
            dct["set_args"] = "%(el_name)s" % dct
        if self.make_package:
            dct["cls_mod_name"] = ""
            if cls_mod_name != self.current_module:
                py_mod_name = self.py_mod_name
                if hasattr(self.types[ft.strip_type(el.type)], "py_mod_name"):
                    py_mod_name = self.types[ft.strip_type(el.type)].py_mod_name
                self.imports.add((py_mod_name + "." + cls_mod_name, cls_name))

        if not isinstance(node, ft.Module) or not self.make_package:
            self.write("@property")
            properties.append(el)
        else:
            dct["el_name_get"] = "get_" + el.name
            dct["el_name_set"] = "set_" + el.name
            dct["self"] = ""
            dct["selfdot"] = ""
            dct["selfcomma"] = ""

        # check for name clashes with pre-existing routines
        if hasattr(node, "procedures"):
            procs = [proc.name for proc in node.procedures]
            if dct["el_name_get"] in procs:
                dct["el_name_get"] += "_"
            if dct["el_name_set"] in procs:
                dct["el_name_set"] += "_"

        # Compute shortened getter/setter names
        dct['subroutine_name_get'] = shorten_long_name('%(prefix)s%(scope_name)s__get__%(el_name)s' % dct)
        dct['subroutine_name_set'] = shorten_long_name('%(prefix)s%(scope_name)s__set__%(el_name)s' % dct)

        self.write("def %(el_name_get)s(%(self)s):" % dct)
        self.indent()
        self.write(self._format_doc_string(el))
        if isinstance(node, ft.Module) and self.make_package:
            self.write("global %(el_name)s" % dct)
        self.write(
            """%(el_name)s_handle = %(mod_name)s.%(subroutine_name_get)s(%(handle)s)
if tuple(%(el_name)s_handle) in %(selfdot)s_objs:
    %(el_name)s = %(selfdot)s_objs[tuple(%(el_name)s_handle)]
else:
    %(el_name)s = %(cls_mod_name)s%(cls_name)s.from_handle(%(el_name)s_handle)
    %(selfdot)s_objs[tuple(%(el_name)s_handle)] = %(el_name)s
return %(el_name)s"""
            % dct
        )
        self.dedent()
        self.write()

        if "parameter" not in el.attributes:
            if not isinstance(node, ft.Module) or not self.make_package:
                self.write("@%(el_name_set)s.setter" % dct)
            self.write(
                """def %(el_name_set)s(%(selfcomma)s%(el_name)s):
    %(el_name)s = %(el_name)s._handle
    %(mod_name)s.%(subroutine_name_set)s(%(set_args)s)
    """
                % dct
            )
            self.write()

    def write_sc_array_wrapper(self, node, el, dims, properties):
        # For module-level arrays, we do not pass a handle to the Fortran
        # subroutine (issue #306: sizeof_fortran_t may differ between
        # generation and runtime environments).
        is_module_array = isinstance(node, ft.Module)
        dct = dict(
            orig_name=el.orig_name,
            el_name=el.name,
            el_name_get=el.name,
            el_name_set=el.name,
            mod_name=self.f90_mod_name,
            prefix=self.prefix,
            type_name=node.name,
            scope_name=self._scope_identifier_for(node),
            self="self",
            selfdot="self.",
            selfcomma="self, ",
            doc=self._format_doc_string(el),
            handle="self._handle" if not is_module_array else "",
        )

        if not is_module_array or not self.make_package:
            self.write("@property")
            properties.append(el)
        else:
            dct["el_name_get"] = "get_array_" + el.name
            dct["el_name_set"] = "set_array_" + el.name
            dct["self"] = ""
            dct["selfdot"] = ""
            dct["selfcomma"] = ""

        self.write("def %(el_name_get)s(%(self)s):" % dct)
        self.indent()
        self.write(self._format_doc_string(el))
        if is_module_array and self.make_package:
            self.write("global %(el_name)s" % dct)
            node.array_initialisers.append(dct["el_name_get"])

        dct["subroutine_name"] = shorten_long_name(
            "%(prefix)s%(scope_name)s__array__%(el_name)s" % dct
        )

        if is_module_array:
            # Module-level arrays: call without handle argument
            self.write(
                """array_ndim, array_type, array_shape, array_handle = \
    %(mod_name)s.%(subroutine_name)s()
if array_handle == 0:
    %(el_name)s = None
else:
    array_shape = list(array_shape[:array_ndim])
    array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
    if array_hash in %(selfdot)s_arrays:
        %(el_name)s = %(selfdot)s_arrays[array_hash]
    else:
        %(el_name)s = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
        %(selfdot)s_arrays[array_hash] = %(el_name)s
return %(el_name)s"""
                % dct
            )
        else:
            # Type member arrays: pass handle argument
            self.write(
                """array_ndim, array_type, array_shape, array_handle = \
    %(mod_name)s.%(subroutine_name)s(%(handle)s)
array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
%(el_name)s = %(selfdot)s_arrays.get(array_hash)
if %(el_name)s is not None:
    # Validate cached array: check data pointer matches current handle (issue #222)
    # Arrays can be deallocated and reallocated at same address, invalidating cache
    if %(el_name)s.ctypes.data != array_handle:
        %(el_name)s = None
if %(el_name)s is None:
    try:
        %(el_name)s = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                %(handle)s,
                                %(mod_name)s.%(subroutine_name)s)
    except TypeError:
        %(el_name)s = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
    %(selfdot)s_arrays[array_hash] = %(el_name)s
return %(el_name)s"""
                % dct
            )
        self.dedent()
        self.write()
        if not isinstance(node, ft.Module) or not self.make_package:
            self.write("@%(el_name)s.setter" % dct)
        if dct["selfdot"]:
            self.write(
                """def %(el_name_set)s(%(selfcomma)s%(el_name)s):
    %(selfdot)s%(el_name)s[...] = %(el_name)s
"""
                % dct
            )
        else:
            self.write(
                """def %(el_name_set)s(%(selfcomma)s%(el_name)s):
    globals()['%(selfdot)s%(el_name)s'][...] = %(el_name)s
"""
                % dct
            )
        self.write()

        if isinstance(node, ft.Module) and dct.get("selfdot"):
            forward_name = "set_array_%s" % el.name
            self.write(f"def {forward_name}(self, value):")
            self.indent()
            self.write(f"self.{el.name}[...] = value")
            self.dedent()
            self.write()
            getter_name = "get_array_%s" % el.name
            self.write(f"def {getter_name}(self):")
            self.indent()
            self.write(f"return self.{el.name}")
            self.dedent()
            self.write()

    def write_dt_array_wrapper(self, node, el, dims):
        if el.type.startswith(("type", "class")) and len(ft.Argument.split_dimensions(dims)) != 1:
            return

        func_name = "init_array_%s" % el.name
        node.dt_array_initialisers.append(func_name)
        cls_name = normalise_class_name(ft.strip_type(el.type), self.class_names)
        mod_name = self.types[ft.strip_type(el.type)].mod_name
        cls_mod_name = self.py_mod_names.get(mod_name, mod_name)

        dct = dict(
            el_name=el.name,
            func_name=func_name,
            mod_name=node.name,
            scope_name=self._scope_identifier_for(node),
            type_name=ft.strip_type(el.type).lower(),
            f90_mod_name=self.f90_mod_name,
            prefix=self.prefix,
            self="self",
            selfdot="self.",
            parent="self",
            doc=self._format_doc_string(el),
            cls_name=cls_name,
            cls_mod_name=normalise_class_name(cls_mod_name, self.class_names) + ".",
        )

        is_module_array = isinstance(node, ft.Module)
        if is_module_array:
            dct["parent"] = "f90wrap.runtime.empty_type"
            if self.make_package:
                dct["selfdot"] = ""
                dct["self"] = ""
        if self.make_package:
            dct["cls_mod_name"] = ""
            if cls_mod_name != self.current_module:
                py_mod_name = self.py_mod_name
                if hasattr(self.types[ft.strip_type(el.type)], "py_mod_name"):
                    py_mod_name = self.types[ft.strip_type(el.type)].py_mod_name
                self.imports.add((py_mod_name + "." + cls_mod_name, cls_name))

        self.write("def %(func_name)s(%(self)s):" % dct)
        self.indent()
        if is_module_array and self.make_package:
            self.write("global %(el_name)s" % dct)

        dct["getitem_name"] = shorten_long_name(
            "%(prefix)s%(scope_name)s__array_getitem__%(el_name)s" % dct
        )
        dct["setitem_name"] = shorten_long_name(
            "%(prefix)s%(scope_name)s__array_setitem__%(el_name)s" % dct
        )
        dct["len_name"] = shorten_long_name(
            "%(prefix)s%(scope_name)s__array_len__%(el_name)s" % dct
        )
        # Module-level arrays do not pass handle to Fortran functions (issue #306)
        dct["module_level"] = "True" if is_module_array else "False"

        # Polymorphic object (class) without assignment method cannot not have setitem
        if el.type.startswith("class") and "has_assignment" not in self.types[ft.strip_type(el.type)].attributes:
            self.write(
                """%(selfdot)s%(el_name)s = f90wrap.runtime.FortranDerivedTypeArray(%(parent)s,
                                    %(f90_mod_name)s.%(getitem_name)s,
                                    None,
                                    %(f90_mod_name)s.%(len_name)s,
                                    %(doc)s, %(cls_mod_name)s%(cls_name)s,
                                    module_level=%(module_level)s)"""
                % dct
            )
        else:
            self.write(
                """%(selfdot)s%(el_name)s = f90wrap.runtime.FortranDerivedTypeArray(%(parent)s,
                                    %(f90_mod_name)s.%(getitem_name)s,
                                    %(f90_mod_name)s.%(setitem_name)s,
                                    %(f90_mod_name)s.%(len_name)s,
                                    %(doc)s, %(cls_mod_name)s%(cls_name)s,
                                    module_level=%(module_level)s)"""
                % dct
            )
        self.write("return %(selfdot)s%(el_name)s" % dct)
        self.dedent()
        self.write()

    def write_type_checks(self, node):
        # This adds tests that checks data types and dimensions
        # to ensure either the correct version of an interface is used
        # either an exception is returned
        for arg in self._filtered_arguments:
            # Check if optional argument is being passed
            if "optional" in arg.attributes:
                self.write("if {0} is not None:".format(arg.py_name))
                self.indent()

            ft_array_dim_list = list(
                filter(lambda x: x.startswith("dimension"), arg.attributes)
            )
            if ft_array_dim_list:
                if ":" in ft_array_dim_list[0]:
                    ft_array_dim = ft_array_dim_list[0].count(",") + 1
                else:
                    ft_array_dim = -1
            else:
                ft_array_dim = 0

            # Checks for derived types
            if arg.type.startswith(("type", "class")):
                cls_mod_name = self.types[ft.strip_type(arg.type)].mod_name
                cls_mod_name = self.py_mod_names.get(cls_mod_name, cls_mod_name)
                py_mod_name = self.py_mod_name
                if hasattr(self.types[ft.strip_type(arg.type)], "py_mod_name"):
                    py_mod_name = self.types[ft.strip_type(arg.type)].py_mod_name
                cls_name = normalise_class_name(
                    ft.strip_type(arg.type), self.class_names
                )
                self.write(
                    "if not isinstance({0}, {1}.{2}) :".format(
                        arg.py_name, cls_mod_name, cls_name
                    )
                )
                self.indent()
                self.write(f"msg = f\"Expecting '{{{cls_mod_name}.{cls_name}}}' but got '{{type({arg.py_name})}}'\"")
                self.write(f"raise TypeError(msg)")
                self.dedent()

                if self.make_package or py_mod_name != self.py_mod_name:
                    self.imports.add((py_mod_name, cls_mod_name))
            else:
                # Checks for Numpy array dimension and types
                # It will fail for types that are not in the kind map
                # Good enough for now if it works on standrad types
                try:
                    array_type = ft.fortran_array_type(arg.type, self.kind_map)
                    pytype = ft.f2numpy_type(arg.type, self.kind_map)
                except RuntimeError:
                    continue

                self.write(
                    "if isinstance({0},(numpy.ndarray, numpy.generic)):".format(
                        arg.py_name
                    )
                )
                self.indent()

                convertible_types = [
                    np.short,
                    np.ushort,
                    np.int32,
                    np.uintc,
                    np.int64,
                    np.uint,
                    np.longlong,
                    np.ulonglong,
                    np.float16,
                    np.float32,
                    np.float64,
                    np.longdouble,
                ]
                if ft_array_dim == 0 and "intent(in)" in arg.attributes:
                    self.write(
                        "if not interface_call and {0}.dtype.num in {{{1}}}:".format(
                            arg.py_name,
                            ", ".join(
                                [str(atype().dtype.num) for atype in convertible_types]
                            ),
                        )
                    )
                    self.indent()
                    self.write("{0} = {0}.astype('{1}')".format(arg.py_name, pytype))
                    self.dedent()

                # Allow fortran character to match python ubyte, unicode_ or string_
                if array_type == np.ubyte().dtype.num:
                    str_types = {
                        np.ubyte().dtype.num,
                        np.bytes_().dtype.num,
                        np.str_().dtype.num,
                    }
                    str_types = {str(num) for num in str_types}
                    # Python char array have one supplementary dimension
                    # https://stackoverflow.com/questions/41864984/how-to-pass-array-of-strings-to-fortran-subroutine-using-f2py
                    if ft_array_dim > 0:
                        str_dims = {ft_array_dim, ft_array_dim + 1}
                    else:
                        str_dims = {
                            ft_array_dim,
                        }
                    str_dims = {str(num) for num in str_dims}
                    if ft_array_dim == -1:
                        self.write(
                            "if {0}.dtype.num not in {{{1}}}:".format(
                                arg.py_name, ",".join(str_types)
                            )
                        )
                    else:
                        self.write(
                            "if {0}.ndim not in {{{1}}} or {0}.dtype.num not in {{{2}}}:".format(
                                arg.py_name, ",".join(str_dims), ",".join(str_types)
                            )
                        )
                elif ft_array_dim == -1:
                    self.write(
                        "if {0}.dtype.num != {1}:".format(arg.py_name, array_type)
                    )
                else:
                    self.write(
                        "if {0}.ndim != {1} or {0}.dtype.num != {2}:".format(
                            arg.py_name, str(ft_array_dim), array_type
                        )
                    )

                self.indent()
                self.write(
                    "raise TypeError(\"Expecting '{0}' (code '{1}')\"\n"
                    "\" with dim '{2}' but got '%s' (code '%s') with dim '%s'\"\n"
                    "%({3}.dtype, {3}.dtype.num, {3}.ndim))".format(
                        ft.f2py_type(arg.type),
                        array_type,
                        str(ft_array_dim),
                        arg.py_name,
                    )
                )
                self.dedent()
                self.dedent()
                if ft_array_dim == 0:
                    self.write(
                        "elif not isinstance({0},{1}):".format(
                            arg.py_name, ft.f2py_type(arg.type)
                        )
                    )
                    self.indent()
                    self.write(
                        "raise TypeError(\"Expecting '{0}' but got '%s'\"%type({1}))".format(
                            ft.f2py_type(arg.type), arg.py_name
                        )
                    )
                    self.dedent()
                else:
                    self.write("else:")
                    self.indent()
                    self.write(
                        "raise TypeError(\"Expecting numpy array but got '%s'\"%type({0}))".format(
                            arg.py_name
                        )
                    )
                    self.dedent()

            if "optional" in arg.attributes:
                self.dedent()

    def _format_pytype_str(self, node_or_arg):
        """
        Format Python type string for an Argument or Element.

        Handles special cases like logical arrays that need int32 instead of bool.
        """
        pytype = ft.f2py_type(node_or_arg.type, node_or_arg.attributes)
        if pytype in ["float", "int", "complex"]:
            # This allows to specify size, ex: 32 bit, 64 bit
            pytype = ft.f2numpy_type(node_or_arg.type, self.kind_map)
        elif pytype == "bool array":
            # Logical arrays require int32 in f2py (Fortran logical is 4 bytes,
            # but NumPy bool is 1 byte). See issue #307.
            pytype = "int32 array"
        elif pytype == "bool":
            # Scalar logical also maps to int32 for consistency
            # (though less critical since scalars auto-convert)
            if "dimension" not in str(node_or_arg.attributes):
                pytype = "bool"  # Keep scalar bool as-is
        return pytype

    def _format_doc_string(self, node):
        """
        Generate Python docstring from Fortran docstring and call signature
        """

        def _format_line_no(lineno):
            """
            Format Fortran source code line numbers

            FIXME could link to source repository (e.g. github)
            """
            if isinstance(lineno, slice):
                return "lines %d-%d" % (lineno.start, lineno.stop - 1)
            else:
                return "line %d" % lineno

        doc = node.doc[:]  # incoming docstring from Fortran source
        # doc can also be empty
        try:
            if (
                doc and doc[-1][-1] != "\n"
            ):  # Short summary and extended summary have a trailing newline
                doc.append("")
        except IndexError:
            pass
        doc.append(self._format_call_signature(node))
        doc.append("Defined at %s %s" % (node.filename, _format_line_no(node.lineno)))

        if isinstance(node, ft.Procedure):
            # For procedures, write parameters and return values in numpydoc format
            doc.append("")
            # Input parameters
            for i, arg in enumerate(self._filtered_arguments):
                pytype = self._format_pytype_str(arg)
                if i == 0:
                    doc.append("Parameters")
                    doc.append("----------")
                arg_doc = "%s : %s\n%s%s" % (
                    arg.name,
                    pytype,
                    self._indent,
                    arg.doxygen,
                )
                doc.append(arg_doc.strip(", \n%s" % self._indent))
                if arg.doc:
                    for d in arg.doc:
                        doc.append("%s%s" % (self._indent, d))
                    doc.append("")

            if isinstance(node, ft.Function):
                for i, arg in enumerate(self._filtered_ret_val):
                    pytype = self._format_pytype_str(arg)
                    if i == 0:
                        if doc[-1] != "":
                            doc.append("")
                        doc.append("Returns")
                        doc.append("-------")
                    arg_doc = "%s : %s\n%s%s" % (
                        arg.name,
                        pytype,
                        self._indent,
                        arg.doxygen,
                    )
                    doc.append(arg_doc.strip(", \n%s" % self._indent))
                    if arg.doc:
                        for d in arg.doc:
                            doc.append("%s%s" % (self._indent, d))
                        doc.append("")
        elif isinstance(node, ft.Interface):
            # for interfaces, list the components
            doc.append("")
            doc.append("Overloaded interface containing the following procedures:")
            for proc in node.procedures:
                doc.append(
                    "  %s"
                    % (hasattr(proc, "method_name") and proc.method_name or proc.name)
                )

        # Escape backslashes in docstrings to avoid SyntaxWarnings
        doc = [line.replace('\\', '\\\\') for line in doc]
        return "\n".join(['"""'] + doc + ['"""'])


    def _format_call_signature(self, node):
        if isinstance(node, ft.Procedure):
            sig = ""
            if isinstance(node, ft.Function):
                sig += ", ".join(ret_val.py_name for ret_val in self._filtered_ret_val)
                sig += " = "
            if "constructor" in node.attributes:
                sig += node.type_name.title()
            elif "destructor" in node.attributes:
                return "Destructor for class %s" % node.type_name.title()
            else:
                if hasattr(node, "method_name"):
                    sig += node.method_name
                else:
                    sig += node.name
            sig += "("
            had_optional = False
            for i, arg in enumerate(self._filtered_arguments):
                if not had_optional and "optional" in arg.attributes:
                    sig += "["
                    had_optional = True
                if i != 0:
                    sig += ", "
                sig += arg.py_name
            if had_optional:
                sig += "]"
            sig += ")"
            rex = re.compile(r"\s+")  # collapse multiple whitespace
            sig = rex.sub(" ", sig)
            return sig
        elif isinstance(node, ft.Module):
            return "Module %s" % node.name
        elif isinstance(node, ft.Element):
            return "Element %s ftype=%s pytype=%s" % (
                node.name,
                node.type,
                self._format_pytype_str(node),
            )
        elif isinstance(node, ft.Interface):
            if hasattr(node, "method_name"):
                name = node.method_name
            else:
                name = node.name
            return "%s(*args, **kwargs)" % name
        else:
            return str(node)
