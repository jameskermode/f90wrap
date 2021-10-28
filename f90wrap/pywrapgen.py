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

from f90wrap.transform import ArrayDimensionConverter
from f90wrap.transform import shorten_long_name
from f90wrap import fortran as ft
from f90wrap import codegen as cg

log = logging.getLogger(__name__)

def py_arg_value(arg):
    # made global from PythonWrapperGenerator.visit_Procedure so that other functions can use it
    if 'optional' in arg.attributes or arg.value is None:
        return '=None'
    else:
        return ''


def normalise_class_name(name, name_map):
    return name_map.get(name.lower(), name.title())


def format_call_signature(node):
    if isinstance(node, ft.Procedure):
        sig = ''
        if isinstance(node, ft.Function):
            sig += ', '.join(ret_val.py_name for ret_val in node.ret_val)
            sig += ' = '
        if 'constructor' in node.attributes:
            sig += node.type_name.title()
        elif 'destructor' in node.attributes:
            return 'Destructor for class %s' % node.type_name.title()
        else:
            if hasattr(node, 'method_name'):
                sig += node.method_name
            else:
                sig += node.name
        sig += '('
        had_optional = False
        for i, arg in enumerate(node.arguments):
            if not had_optional and 'optional' in arg.attributes:
                sig += '['
                had_optional = True
            if i != 0:
                sig += ', '
            sig += arg.py_name
        if had_optional:
            sig += ']'
        sig += ')'
        rex = re.compile(r'\s+')  # collapse multiple whitespace
        sig = rex.sub(' ', sig)
        return sig
    elif isinstance(node, ft.Module):
        return 'Module %s' % node.name
    elif isinstance(node, ft.Element):
        return ('Element %s ftype=%s pytype=%s' %
                (node.name, node.type,
                 ft.f2py_type(node.type)))
    elif isinstance(node, ft.Interface):
        if hasattr(node, 'method_name'):
            name = node.method_name
        else:
            name = node.name
        return '%s(*args, **kwargs)' % name
    else:
        return str(node)


def format_doc_string(node):
    """
    Generate Python docstring from Fortran docstring and call signature
    """

    def _format_line_no(lineno):
        """
        Format Fortran source code line numbers

        FIXME could link to source repository (e.g. github)
        """
        if isinstance(lineno, slice):
            return 'lines %d-%d' % (lineno.start, lineno.stop - 1)
        else:
            return 'line %d' % lineno

    doc = [format_call_signature(node), '']
    doc.append('')
    doc.append('Defined at %s %s' % (node.filename, _format_line_no(node.lineno)))

    if isinstance(node, ft.Procedure):
        # For procedures, write parameters and return values in numpydoc format
        doc.append('')
        # Input parameters
        for i, arg in enumerate(node.arguments):
            pytype = ft.f2py_type(arg.type, arg.attributes)
            if i == 0:
                doc.append("Parameters")
                doc.append("----------")
            arg_doc = "%s : %s, %s" % (arg.name, pytype, arg.doxygen)
            doc.append(arg_doc.strip(', '))
            if arg.doc:
                for d in arg.doc:
                    doc.append("\t%s" % d)
                doc.append("")

        if isinstance(node, ft.Function):
            for i, arg in enumerate(node.ret_val):
                pytype = ft.f2py_type(arg.type, arg.attributes)
                if i == 0:
                    doc.append("")
                    doc.append("Returns")
                    doc.append("-------")
                arg_doc = "%s : %s, %s" % (arg.name, pytype, arg.doxygen)
                doc.append(arg_doc.strip(', '))
                if arg.doc:
                    for d in arg.doc:
                        doc.append("\t%s" % d)
                    doc.append("")
    elif isinstance(node, ft.Interface):
        # for interfaces, list the components
        doc.append('')
        doc.append('Overloaded interface containing the following procedures:')
        for proc in node.procedures:
            doc.append('  %s' % (hasattr(proc, 'method_name')
                                 and proc.method_name or proc.name))

    doc += [''] + node.doc[:]  # incoming docstring from Fortran source

    return '\n'.join(['"""'] + doc + ['"""'])


class PythonWrapperGenerator(ft.FortranVisitor, cg.CodeGenerator):
    def __init__(self, prefix, mod_name, types, f90_mod_name=None,
                 make_package=False, kind_map=None, init_file=None,
                 py_mod_names=None, class_names=None, max_length=None,
                 type_check=False):
        if max_length is None:
            max_length = 80
        cg.CodeGenerator.__init__(self, indent=' ' * 4,
                                  max_length=max_length,
                                  continuation='\\',
                                  comment='#')
        ft.FortranVisitor.__init__(self)
        self.prefix = prefix
        self.py_mod_name = mod_name
        self.py_mod_names = py_mod_names
        self.class_names = class_names
        if f90_mod_name is None:
            f90_mod_name = '_' + mod_name
        self.f90_mod_name = f90_mod_name
        self.types = types
        self.imports = set()
        self.make_package = make_package
        if kind_map is None:
            kind_map = {}
        self.kind_map = kind_map
        self.init_file = init_file
        self.type_check = type_check

    def write_imports(self, insert=0):
        default_imports = [(self.f90_mod_name, None),
                           ('f90wrap.runtime', None),
                           ('logging', None),
                           ('numpy', None)]
        imp_lines = ['from __future__ import print_function, absolute_import, division']
        for (mod, symbol) in default_imports + list(self.imports):
            if symbol is None:
                imp_lines.append('import %s' % mod)
            elif isinstance(symbol, tuple):
                imp_lines.append('from %s import %s' % (mod, ', '.join(symbol)))
            else:
                imp_lines.append('from %s import %s' % (mod, symbol))
        imp_lines += ['\n']
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
                self.imports.add((self.py_mod_name + '.' +
                                  self.py_mod_names.get(py_mod, py_mod), None))
        self.write_imports(0)

        if self.make_package:
            py_wrapper_file = open(os.path.join(self.py_mod_name, '__init__.py'), 'w')
        else:
            py_wrapper_file = open('%s.py' % self.py_mod_name, 'w')
        py_wrapper_file.write(str(self))
        if self.init_file is not None:
            py_wrapper_file.write(open(self.init_file).read())
        py_wrapper_file.close()

    def visit_Module(self, node):
        log.info('PythonWrapperGenerator visiting module %s' % node.name)
        cls_name = normalise_class_name(node.name, self.class_names)
        node.array_initialisers = []
        node.dt_array_initialisers = []
        self.current_module = self.py_mod_names.get(node.name, node.name)

        if self.make_package:
            self.code = []
            self.write(format_doc_string(node))
        else:
            self.write('class %s(f90wrap.runtime.FortranModule):' % cls_name)
            self.indent()
            self.write(format_doc_string(node))

            if len(node.elements) == 0 and len(node.types) == 0 and len(node.procedures) == 0:
                self.write('pass')

        index = len(self.code)  # save position to insert import lines

        self.generic_visit(node)

        properties = []  # Collect list of properties for a __repr__()
        for el in node.elements:
            dims = list(filter(lambda x: x.startswith('dimension'), el.attributes))
            if len(dims) == 0:  # proper scalar type (normal or derived)
                if el.type.startswith('type'):
                    self.write_dt_wrappers(node, el, properties)
                else:
                    self.write_scalar_wrappers(node, el, properties)
            elif el.type.startswith('type'):  # array of derived types
                self.write_dt_array_wrapper(node, el, dims[0])
            else:
                self.write_sc_array_wrapper(node, el, dims[0], properties)
        self.write_repr(node, properties)

        # insert import statements at the beginning of module
        if self.make_package:
            index = self.write_imports(index)
            self.writelines(['_arrays = {}', '_objs = {}', '\n'],
                            insert=index)
            self.write()

        if self.make_package:
            self.write('_array_initialisers = [%s]' % (', '.join(node.array_initialisers)))
        self.write('_dt_array_initialisers = [%s]' % (', '.join(node.dt_array_initialisers)))
        self.write()

        # FIXME - make this less ugly, e.g. by generating code for each array
        if self.make_package:
            self.write('''try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module "%s".')
''' % node.name)
            self.write()
            self.write('''for func in _dt_array_initialisers:
    func()
            ''')
            if len(self.code) > 0:
                py_mod_name = self.py_mod_names.get(node.name, node.name)
                py_wrapper_file = open(os.path.join(self.py_mod_name, py_mod_name + '.py'), 'w')
                py_wrapper_file.write(str(self))
                py_wrapper_file.close()
                self.py_mods.append(node.name)
            self.code = []
        else:
            self.dedent()  # finish the FortranModule class
            self.write()
            # instantise the module class
            self.write('%s = %s()' % (node.name, cls_name))
            self.write()

        self.current_module = None

    def write_constructor(self, node):
        handle_arg = ft.Argument(name='handle',
                                 filename=node.filename,
                                 doc=['Opaque reference to existing derived type instance'],
                                 lineno=node.lineno,
                                 attributes=['intent(in)', 'optional'],
                                 type='integer')
        handle_arg.py_name = 'handle'

        # special case for constructors: return value is 'self' argument,
        # plus we add an extra optional argument
        args = node.arguments + [handle_arg]

        dct = dict(func_name=node.name,
                   prefix=self.prefix,
                   mod_name=self.f90_mod_name,
                   py_arg_names=', '.join(['%s%s' % (arg.py_name,
                                                     'optional' in arg.attributes and '=None' or '')
                                           for arg in args]),
                   f90_arg_names=', '.join(['%s=%s' % (arg.name, arg.py_value) for arg in node.arguments]))

        self.write("def __init__(self, %(py_arg_names)s):" % dct)
        self.indent()
        self.write(format_doc_string(node))
        for arg in node.arguments:
            if 'optional' in arg.attributes and '._handle' in arg.py_value:
                dct['f90_arg_names'] = dct['f90_arg_names'].replace(arg.py_value,
                                                                    ('(None if %(arg_py_name)s is None else %('
                                                                     'arg_py_name)s._handle)') %
                                                                    {'arg_py_name': arg.py_name})
        self.write('f90wrap.runtime.FortranDerivedType.__init__(self)')

        self.write('result = %(mod_name)s.%(prefix)s%(func_name)s(%(f90_arg_names)s)' % dct)
        self.write('self._handle = result[0] if isinstance(result, tuple) else result')
        self.dedent()
        self.write()

    def write_classmethod(self, node):

        dct = dict(func_name=node.name,
                   method_name=hasattr(node, 'method_name') and node.method_name or node.name,
                   prefix=self.prefix,
                   mod_name=self.f90_mod_name,
                   py_arg_names=', '.join([arg.py_name + py_arg_value(arg) for arg in node.arguments]),
                   f90_arg_names=', '.join(['%s=%s' % (arg.name, arg.py_value) for arg in node.arguments]),
                   call='')

        dct['call'] = 'result = '
        for arg in node.arguments:
            if 'optional' in arg.attributes and '._handle' in arg.py_value:
                dct['f90_arg_names'] = dct['f90_arg_names'].replace(arg.py_value,
                                                                    ('None if %(arg_py_name)s is None else %('
                                                                     'arg_py_name)s._handle') %
                                                                    {'arg_py_name': arg.py_name})
        call_line = '%(call)s%(mod_name)s.%(prefix)s%(func_name)s(%(f90_arg_names)s)' % dct

        self.write('@classmethod')
        self.write("def %(method_name)s(cls, %(py_arg_names)s):" % dct)
        self.indent()
        self.write(format_doc_string(node))
        self.write('bare_class = cls.__new__(cls)')
        self.write('f90wrap.runtime.FortranDerivedType.__init__(bare_class)')

        self.write(call_line)

        self.write('bare_class._handle = result[0] if isinstance(result, tuple) else result')
        self.write('return bare_class')

        self.dedent()
        self.write()

    def write_destructor(self, node):
        dct = dict(func_name=node.name,
                   prefix=self.prefix,
                   mod_name=self.f90_mod_name,
                   py_arg_names=', '.join(['%s%s' % (arg.py_name,
                                                     'optional' in arg.attributes and '=None' or '')
                                           for arg in node.arguments]),
                   f90_arg_names=', '.join(['%s=%s' % (arg.name, arg.py_value) for arg in node.arguments]))
        self.write("def __del__(%(py_arg_names)s):" % dct)
        self.indent()
        self.write(format_doc_string(node))
        self.write('if self._alloc:')
        self.indent()
        self.write('%(mod_name)s.%(prefix)s%(func_name)s(%(f90_arg_names)s)' % dct)
        self.dedent()
        self.dedent()
        self.write()

    def visit_Procedure(self, node):

        log.info('PythonWrapperGenerator visiting routine %s' % node.name)
        if 'classmethod' in node.attributes:
            self.write_classmethod(node)
        elif 'constructor' in node.attributes:
            self.write_constructor(node)
        elif 'destructor' in node.attributes:
            self.write_destructor(node)
        else:
            dct = dict(func_name=node.name,
                       method_name=hasattr(node, 'method_name') and node.method_name or node.name,
                       prefix=self.prefix,
                       mod_name=self.f90_mod_name,
                       py_arg_names=', '.join([arg.py_name + py_arg_value(arg) for arg in node.arguments]),
                       f90_arg_names=', '.join(['%s=%s' % (arg.name, arg.py_value) for arg in node.arguments]),
                       call='')

            if isinstance(node, ft.Function):
                dct['result'] = ', '.join([ret_val.name for ret_val in node.ret_val])
                dct['call'] = '%(result)s = ' % dct

            if not self.make_package and node.mod_name is not None and node.type_name is None:
                # procedures outside of derived types become static methods
                self.write('@staticmethod')
            self.write("def %(method_name)s(%(py_arg_names)s):" % dct)
            self.indent()
            self.write(format_doc_string(node))

            if self.type_check:
                self.write_type_checks(node)

            for arg in node.arguments:
                if 'optional' in arg.attributes and '._handle' in arg.py_value:
                    dct['f90_arg_names'] = dct['f90_arg_names'].replace(arg.py_value,
                                                                        (
                                                                            'None if %(arg_py_name)s is None else %(arg_py_name)s._handle') %
                                                                        {'arg_py_name': arg.py_name})
            call_line = '%(call)s%(mod_name)s.%(prefix)s%(func_name)s(%(f90_arg_names)s)' % dct
            self.write(call_line)

            if isinstance(node, ft.Function):
                # convert any derived type return values to Python objects
                for ret_val in node.ret_val:
                    if ret_val.type.startswith('type'):
                        cls_name = normalise_class_name(ft.strip_type(ret_val.type), self.class_names)
                        cls_name = self.py_mod_name + '.' + cls_name
                        cls_name = 'f90wrap.runtime.lookup_class("%s")' % cls_name
                        cls_mod_name = self.types[ft.strip_type(ret_val.type)].mod_name
                        cls_mod_name = self.py_mod_names.get(cls_mod_name, cls_mod_name)
                        # if self.make_package:
                        #     if cls_mod_name != self.current_module:
                        #         self.imports.add((self.py_mod_name + '.' + cls_mod_name, cls_name))
                        # else:
                        #     cls_name = cls_mod_name + '.' + cls_name
                        self.write('%s = %s.from_handle(%s, alloc=True)' %
                                   (ret_val.name, cls_name, ret_val.name))
                self.write('return %(result)s' % dct)

            self.dedent()
            self.write()

    def visit_Interface(self, node):
        log.info('PythonWrapperGenerator visiting interface %s' % node.name)

        # first output all the procedures within the interface
        self.generic_visit(node)
        cls_name = None
        if node.type_name is not None:
            cls_name = normalise_class_name(ft.strip_type(node.type_name),
                                            self.class_names)
        proc_names = []
        for proc in node.procedures:
            proc_name = ''
            if not self.make_package and hasattr(proc, 'mod_name'):
                proc_name += normalise_class_name(proc.mod_name, self.class_names) + '.'
            elif cls_name is not None:
                proc_name += cls_name + '.'
            if hasattr(proc, 'method_name'):
                proc_name += proc.method_name
            else:
                proc_name += proc.name
            proc_names.append(proc_name)

        dct = dict(intf_name=node.method_name,
                   proc_names='[' + ', '.join(proc_names) + ']')
        if not self.make_package:
            # procedures outside of derived types become static methods
            self.write('@staticmethod')
        self.write('def %(intf_name)s(*args, **kwargs):' % dct)
        self.indent()
        self.write(format_doc_string(node))
        # try to call each in turn until no TypeError raised
        self.write('for proc in %(proc_names)s:' % dct)
        self.indent()
        self.write('try:')
        self.indent()
        self.write('return proc(*args, **kwargs)')
        self.dedent()
        self.write('except TypeError:')
        self.indent()
        self.write('continue')
        self.dedent()
        self.dedent()
        self.write()

        if self.type_check:
            self.write('argTypes=[]')
            self.write('for arg in args:')
            self.indent()
            self.write('try:')
            self.indent()
            self.write('argTypes.append("%s: dims \'%s\', type \'%s\'"%(str(type(arg)),'
                        'arg.ndim, arg.dtype))')
            self.dedent()
            self.write('except AttributeError:')
            self.indent()
            self.write('argTypes.append(str(type(arg)))')
            self.dedent()
            self.dedent()

            self.write('raise TypeError("Not able to call a version of "')
            self.indent()
            self.write('"\'%(intf_name)s\' compatible with the provided args:"' % dct)
            self.write('"\\n%s\\n"%"\\n".join(argTypes))')
            self.dedent()
        self.dedent()
        self.write()

    def visit_Type(self, node):
        log.info('PythonWrapperGenerator visiting type %s' % node.name)
        node.dt_array_initialisers = []
        cls_name = normalise_class_name(node.name, self.class_names)
        self.write('@f90wrap.runtime.register_class("%s.%s")' % (self.py_mod_name, cls_name))
        self.write('class %s(f90wrap.runtime.FortranDerivedType):' % cls_name)
        self.indent()
        self.write(format_doc_string(node))
        self.generic_visit(node)

        properties = []
        for el in node.elements:
            dims = list(filter(lambda x: x.startswith('dimension'), el.attributes))
            if len(dims) == 0:  # proper scalar type (normal or derived)
                if el.type.startswith('type'):
                    self.write_dt_wrappers(node, el, properties)
                else:
                    self.write_scalar_wrappers(node, el, properties)
            elif el.type.startswith('type'):  # array of derived types
                self.write_dt_array_wrapper(node, el, dims[0])
            else:
                self.write_sc_array_wrapper(node, el, dims, properties)
        self.write_repr(node, properties)

        self.write('_dt_array_initialisers = [%s]' % (', '.join(node.dt_array_initialisers)))
        self.write()
        self.dedent()
        self.write()

    def write_scalar_wrappers(self, node, el, properties):
        dct = dict(el_name=el.name,
                   el_orig_name=el.orig_name,
                   el_name_get=el.name,
                   el_name_set=el.name,
                   mod_name=self.f90_mod_name,
                   prefix=self.prefix, type_name=node.name,
                   self='self',
                   selfdot='self.',
                   selfcomma='self, ',
                   handle=isinstance(node, ft.Type) and 'self._handle' or '')

        if hasattr(el, 'py_name'):
            dct['el_name_get'] = el.py_name
            dct['el_name_set'] = el.py_name

        if isinstance(node, ft.Type):
            dct['set_args'] = '%(handle)s, %(el_name_get)s' % dct
        else:
            dct['set_args'] = '%(el_name_get)s' % dct

        if not isinstance(node, ft.Module) or not self.make_package:
            self.write('@property')
            properties.append(el)
        else:
            dct['el_name_get'] = 'get_' + el.name
            dct['el_name_set'] = 'set_' + el.name
            dct['self'] = ''
            dct['selfdot'] = ''
            dct['selfcomma'] = ''

        # check for name clashes with pre-existing routines
        if hasattr(node, 'procedures'):
            procs = [proc.name for proc in node.procedures]
            if dct['el_name_get'] in procs:
                dct['el_name_get'] += '_'
            if dct['el_name_set'] in procs:
                dct['el_name_set'] += '_'

        dct['subroutine_name'] = shorten_long_name('%(prefix)s%(type_name)s__get__%(el_name)s' % dct)

        self.write('def %(el_name_get)s(%(self)s):' % dct)
        self.indent()
        self.write(format_doc_string(el))
        self.write('return %(mod_name)s.%(subroutine_name)s(%(handle)s)' % dct)
        self.dedent()
        self.write()
        if 'parameter' in el.attributes and isinstance(node, ft.Module) and self.make_package:
            self.write('%(el_orig_name)s = %(el_name_get)s()' % dct)
            self.write()

        if 'parameter' not in el.attributes:
            if not isinstance(node, ft.Module) or not self.make_package:
                self.write('@%(el_name_get)s.setter' % dct)
            self.write('''def %(el_name_set)s(%(selfcomma)s%(el_name)s):
    %(mod_name)s.%(prefix)s%(type_name)s__set__%(el_name)s(%(set_args)s)
    ''' % dct)
            self.write()

    def write_repr(self, node, properties):
        if len(properties) < 1: return

        self.write('def __str__(self):')

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
        dct = dict(el_name=el.name,
                   el_name_get=el.name,
                   el_name_set=el.name,
                   mod_name=self.f90_mod_name,
                   prefix=self.prefix, type_name=node.name,
                   cls_name=cls_name,
                   cls_mod_name=cls_mod_name + '.',
                   self='self',
                   selfdot='self.',
                   selfcomma='self, ',
                   handle=isinstance(node, ft.Type) and 'self._handle' or '')
        if isinstance(node, ft.Type):
            dct['set_args'] = '%(handle)s, %(el_name)s' % dct
        else:
            dct['set_args'] = '%(el_name)s' % dct
        if self.make_package:
            dct['cls_mod_name'] = ''
            if cls_mod_name != self.current_module:
                self.imports.add((self.py_mod_name + '.' + cls_mod_name, cls_name))

        if not isinstance(node, ft.Module) or not self.make_package:
            self.write('@property')
            properties.append(el)
        else:
            dct['el_name_get'] = 'get_' + el.name
            dct['el_name_set'] = 'set_' + el.name
            dct['self'] = ''
            dct['selfdot'] = ''
            dct['selfcomma'] = ''

        # check for name clashes with pre-existing routines
        if hasattr(node, 'procedures'):
            procs = [proc.name for proc in node.procedures]
            if dct['el_name_get'] in procs:
                dct['el_name_get'] += '_'
            if dct['el_name_set'] in procs:
                dct['el_name_set'] += '_'

        self.write('def %(el_name_get)s(%(self)s):' % dct)
        self.indent()
        self.write(format_doc_string(el))
        if isinstance(node, ft.Module) and self.make_package:
            self.write('global %(el_name)s' % dct)
        self.write('''%(el_name)s_handle = %(mod_name)s.%(prefix)s%(type_name)s__get__%(el_name)s(%(handle)s)
if tuple(%(el_name)s_handle) in %(selfdot)s_objs:
    %(el_name)s = %(selfdot)s_objs[tuple(%(el_name)s_handle)]
else:
    %(el_name)s = %(cls_mod_name)s%(cls_name)s.from_handle(%(el_name)s_handle)
    %(selfdot)s_objs[tuple(%(el_name)s_handle)] = %(el_name)s
return %(el_name)s''' % dct)
        self.dedent()
        self.write()

        if 'parameter' not in el.attributes:
            if not isinstance(node, ft.Module) or not self.make_package:
                self.write('@%(el_name_set)s.setter' % dct)
            self.write('''def %(el_name_set)s(%(selfcomma)s%(el_name)s):
    %(el_name)s = %(el_name)s._handle
    %(mod_name)s.%(prefix)s%(type_name)s__set__%(el_name)s(%(set_args)s)
    ''' % dct)
            self.write()

    def write_sc_array_wrapper(self, node, el, dims, properties):
        dct = dict(el_name=el.name,
                   el_name_get=el.name,
                   el_name_set=el.name,
                   mod_name=self.f90_mod_name,
                   prefix=self.prefix, type_name=node.name,
                   self='self',
                   selfdot='self.',
                   selfcomma='self, ',
                   doc=format_doc_string(el),
                   handle=isinstance(node, ft.Type) and 'self._handle' or 'f90wrap.runtime.empty_handle')

        if not isinstance(node, ft.Module) or not self.make_package:
            self.write('@property')
            properties.append(el)
        else:
            dct['el_name_get'] = 'get_array_' + el.name
            dct['el_name_set'] = 'set_array_' + el.name
            dct['self'] = ''
            dct['selfdot'] = ''
            dct['selfcomma'] = ''

        self.write('def %(el_name_get)s(%(self)s):' % dct)
        self.indent()
        self.write(format_doc_string(el))
        if isinstance(node, ft.Module) and self.make_package:
            self.write('global %(el_name)s' % dct)
            node.array_initialisers.append(dct['el_name_get'])

        dct['subroutine_name'] = shorten_long_name('%(prefix)s%(type_name)s__array__%(el_name)s' % dct)

        self.write("""array_ndim, array_type, array_shape, array_handle = \
    %(mod_name)s.%(subroutine_name)s(%(handle)s)
if array_handle in %(selfdot)s_arrays:
    %(el_name)s = %(selfdot)s_arrays[array_handle]
else:
    %(el_name)s = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                            %(handle)s,
                            %(mod_name)s.%(subroutine_name)s)
    %(selfdot)s_arrays[array_handle] = %(el_name)s
return %(el_name)s""" % dct)
        self.dedent()
        self.write()
        if not isinstance(node, ft.Module) or not self.make_package:
            self.write("@%(el_name)s.setter" % dct)
        self.write("""def %(el_name_set)s(%(selfcomma)s%(el_name)s):
    %(selfdot)s%(el_name)s[...] = %(el_name)s
""" % dct)
        self.write()

    def write_dt_array_wrapper(self, node, el, dims):
        if el.type.startswith('type') and len(ArrayDimensionConverter.split_dimensions(dims)) != 1:
            return

        func_name = 'init_array_%s' % el.name
        node.dt_array_initialisers.append(func_name)
        cls_name = normalise_class_name(ft.strip_type(el.type), self.class_names)
        mod_name = self.types[ft.strip_type(el.type)].mod_name
        cls_mod_name = self.py_mod_names.get(mod_name, mod_name)

        dct = dict(el_name=el.name,
                   func_name=func_name,
                   mod_name=node.name,
                   type_name=ft.strip_type(el.type).lower(),
                   f90_mod_name=self.f90_mod_name,
                   prefix=self.prefix,
                   self='self',
                   selfdot='self.',
                   parent='self',
                   doc=format_doc_string(el),
                   cls_name=cls_name,
                   cls_mod_name=normalise_class_name(cls_mod_name, self.class_names) + '.')

        if isinstance(node, ft.Module):
            dct['parent'] = 'f90wrap.runtime.empty_type'
            if self.make_package:
                dct['selfdot'] = ''
                dct['self'] = ''
        if self.make_package:
            dct['cls_mod_name'] = ''
            if cls_mod_name != self.current_module:
                self.imports.add((self.py_mod_name + '.' + cls_mod_name, cls_name))

        self.write('def %(func_name)s(%(self)s):' % dct)
        self.indent()
        if isinstance(node, ft.Module) and self.make_package:
            self.write('global %(el_name)s' % dct)

        dct['getitem_name'] = shorten_long_name('%(prefix)s%(mod_name)s__array_getitem__%(el_name)s' % dct)
        dct['setitem_name'] = shorten_long_name('%(prefix)s%(mod_name)s__array_setitem__%(el_name)s' % dct)
        dct['len_name'] = shorten_long_name('%(prefix)s%(mod_name)s__array_len__%(el_name)s' % dct)

        self.write('''%(selfdot)s%(el_name)s = f90wrap.runtime.FortranDerivedTypeArray(%(parent)s,
                                %(f90_mod_name)s.%(getitem_name)s,
                                %(f90_mod_name)s.%(setitem_name)s,
                                %(f90_mod_name)s.%(len_name)s,
                                %(doc)s, %(cls_mod_name)s%(cls_name)s)''' % dct)
        self.write('return %(selfdot)s%(el_name)s' % dct)
        self.dedent()
        self.write()

    def write_type_checks(self, node):
        # This adds tests that checks data types and dimensions
        # to ensure either the correct version of an interface is used
        # either an exception is returned
        for arg in node.arguments:
            if 'optional' not in arg.attributes:
                ft_array_dim_list = list(filter(lambda x: x.startswith("dimension"),
                        arg.attributes))
                if ft_array_dim_list:
                    if ':' in ft_array_dim_list[0]:
                        ft_array_dim = ft_array_dim_list[0].count(',')+1
                    else:
                        ft_array_dim = -1
                else:
                    ft_array_dim = 0

                # Checks for derived types
                if (arg.type.startswith('type') or arg.type.startswith('class')):
                    cls_mod_name = self.types[ft.strip_type(arg.type)].mod_name
                    cls_mod_name = self.py_mod_names.get(cls_mod_name, cls_mod_name)

                    cls_name = normalise_class_name(ft.strip_type(arg.type), self.class_names)
                    self.write('if not isinstance({0}, {1}.{2}) :'\
                        .format(arg.py_name, cls_mod_name, cls_name) )
                    self.indent()
                    self.write('raise TypeError')
                    self.dedent()

                    if self.make_package:
                        self.imports.add((self.py_mod_name, cls_mod_name))
                else:
                    # Checks for Numpy array dimension and types
                    # It will fail for types that are not in the kind map
                    # Good enough for now if it works on standrad types
                    try:
                        array_type=ft.fortran_array_type(arg.type, self.kind_map)
                    except RuntimeError:
                        continue

                    py_type = ft.f2py_type(arg.type)

                    # bool are ignored because fortran logical are mapped to integers
                    if py_type not in  ['bool']:
                        self.write('if isinstance({0},(numpy.ndarray, numpy.generic)):'\
                                  .format(arg.py_name))
                        self.indent()
                        if ft_array_dim == -1:
                            self.write('if {0}.dtype.num != {1}:'\
                                    .format(arg.py_name, array_type))
                        else:
                            self.write('if {0}.ndim != {1} or {0}.dtype.num != {2}:'\
                                    .format(arg.py_name, str(ft_array_dim), array_type))

                        self.indent()
                        self.write('raise TypeError')
                        self.dedent()
                        self.dedent()
                        if ft_array_dim == 0:
                            # Do not write checks for unknown types
                            if py_type not in  ['unknown']:
                                self.write('elif not isinstance({0},{1}):'\
                                          .format(arg.py_name,py_type))
                                self.indent()
                                self.write('raise TypeError')
                                self.dedent()
                        else:
                            self.write('else:')
                            self.indent()
                            self.write('raise TypeError')
                            self.dedent()
