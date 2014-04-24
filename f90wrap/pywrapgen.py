# HF XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# HF X
# HF X   f90wrap: F90 to Python interface generator with derived type support
# HF X
# HF X   Copyright James Kermode 2011
# HF X
# HF X   These portions of the source code are released under the GNU General
# HF X   Public License, version 2, http://www.gnu.org/copyleft/gpl.html
# HF X
# HF X   If you would like to license the source code under different terms,
# HF X   please contact James Kermode, james.kermode@gmail.com
# HF X
# HF X   When using this software, please cite the following reference:
# HF X
# HF X   http://www.jrkermode.co.uk/f90wrap
# HF X
# HF XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

import logging
import copy
import re
from f90wrap.fortran import (Fortran, Root, Program, Module, Procedure, Subroutine, Function,
                             Declaration, Element, Argument, Type, Interface,
                             FortranVisitor, FortranTransformer, strip_type)
from f90wrap.codegen import CodeGenerator
import numpy as np

class PythonWrapperGenerator(FortranVisitor, CodeGenerator):

    def __init__(self, prefix, mod_name, types, imports=None, f90_mod_name=None):
        CodeGenerator.__init__(self, indent=' ' * 4,
                               max_length=80,
                               continuation='\\')
        FortranVisitor.__init__(self)
        self.prefix = prefix
        self.py_mod_name = mod_name
        if f90_mod_name is None:
            f90_mod_name = '_'+mod_name
        self.f90_mod_name = f90_mod_name
        self.types = types
        if imports is None:
            imports = [(self.f90_mod_name, None),
                       ('f90wrap.sizeof_fortran_t', 'sizeof_fortran_t'),
                       ('f90wrap.arraydata', 'arraydata'),
                       ('f90wrap.fortrantype', 'fortrantype')]
        self.imports = imports

    def format_doc_string(self, node):
        """
        Generate Python docstring from Fortran docstring and call signature
        """

        def _format_line_no(lineno):
            """
            Format Fortran source code line numbers
            
            FIXME could link to source repository (e.g. github)
            """
            if isinstance(lineno, slice):
                return 'lines %d-%d' % (lineno.start, lineno.stop-1)
            else:
                return 'line %d' % lineno
        
        doc = node.doc[:] # start with incoming docstring from Fortran source
        doc.append(str(node))
        doc.append('Defined at %s %s' % (node.filename, _format_line_no(node.lineno)))

        return '\n'.join(['"""'] + doc + ['"""'])

    def write_imports(self):
        for (mod, alias) in self.imports:
            if alias is None:
                self.write('import %s' % mod)
            else:
                self.write('import %s as %s' % (mod, alias)) 
        self.write()
        self.write('_sizeof_fortran_t = sizeof_fortran_t.sizeof_fortran_t()')
        self.write()

    def visit_Root(self, node):
        """
        Wrap subroutines and functions that are outside of any Fortran modules
        """
        self.code = []
        self.write_imports()
        self.generic_visit(node)
        
        if len(self.code) > 0:
            py_wrapper_file = open('%s.py' % self.py_mod_name, 'w')
            py_wrapper_file.write(str(self))
            py_wrapper_file.close()

    def visit_Module(self, node):
        logging.info('PythonWrapper visiting module %s' % node.name)        
        cls_name = node.name.title()
        self.write('class %s(fortrantype.FortranModule):' % cls_name)
        self.indent()
        self.write(self.format_doc_string(node))
        
        if len(node.elements) == 0 and len(node.types) == 0 and len(node.procedures) == 0:
            self.write('pass')
            
        for el in node.elements:
            dims = filter(lambda x: x.startswith('dimension'), el.attributes)
            if len(dims) == 0:  # proper scalar type (normal or derived)
                if el.type.startswith('type'):
                    self.write_dt_wrappers(node, el) 
                else:
                    self.write_scalar_wrappers(node, el) 
            elif el.type.startswith('type'):  # array of derived types
                self.write_dt_array_wrapper(node, el, dims)
            else:
                self.write_sc_array_wrapper(node, el, dims)

        self.generic_visit(node)

        self.dedent() # finish the FortranModule class
        self.write()
        # instantise the module class
        self.write('%s = %s()' % (node.name, node.name.title()))
        self.write()

    def write_constructor(self, node):
        handle_arg = Argument(name='handle',
                              filename=node.filename,
                              doc=['Opaque reference to existing derived type instance'],
                              lineno=node.lineno,
                              attributes=['intent(in)', 'optional'],
                              type='integer')

        # special case for constructors: return value is 'self' argument,
        # plus we add an extra optional argument
        args = node.ret_val + node.arguments + [handle_arg]
        
        dct = dict(func_name=node.name,
                   prefix=self.prefix,
                   mod_name=self.f90_mod_name,
                   py_arg_names=', '.join(['%s%s' % (arg.name,
                                                     'optional' in arg.attributes and '=None' or '')
                                                     for arg in args ]),
                   f90_arg_names=', '.join(['%s=%s' % (arg.orig_name, arg.value) for arg in node.arguments]))

        self.write("def __init__(%(py_arg_names)s):" % dct)
        self.indent()
        self.write(self.format_doc_string(node))
        self.write('fortrantype.FortranDerivedType.__init__(self)')
        self.write('self._handle = %(mod_name)s.%(prefix)s%(func_name)s(%(f90_arg_names)s)' % dct)
        self.dedent()
        self.write()

    def write_destructor(self, node):
        dct = dict(func_name=node.name,
                   prefix=self.prefix,
                   mod_name=self.f90_mod_name,
                   py_arg_names=', '.join(['%s%s' % (arg.name,
                                                     'optional' in arg.attributes and '=None' or '')
                                                     for arg in node.arguments]),
                   f90_arg_names=', '.join(['%s=%s' % (arg.orig_name, arg.value) for arg in node.arguments]))
        self.write("def __del__(%(py_arg_names)s):" % dct)
        self.indent()
        self.write(self.format_doc_string(node))
        self.write('if self._alloc:')
        self.indent()
        self.write('%(mod_name)s.%(prefix)s%(func_name)s(%(f90_arg_names)s)' % dct)
        self.dedent()
        self.dedent()
        self.write()

    def visit_Procedure(self, node):
        logging.info('PythonWrapper visiting procedure %s' % node.name)
        if 'constructor' in node.attributes:
            self.write_constructor(node)
        elif 'destructor' in node.attributes:
            self.write_destructor(node)
        else:
            dct = dict(func_name=node.name,
                       prefix=self.prefix,
                       mod_name=self.f90_mod_name,
                       py_arg_names=', '.join(['%s%s' % (arg.name,
                                                     arg.value is None and '=None' or '')
                                                     for arg in node.arguments ]),
                       f90_arg_names=', '.join(['%s=%s' % (arg.orig_name, arg.name) for arg in node.arguments]))
                       
            # module procedures become static methods
            if node.type_name is None and node.mod_name is not None:
                self.write('@staticmethod')
            self.write("def %(func_name)s(%(py_arg_names)s):" % dct)
            self.indent()
            self.write(self.format_doc_string(node))
            call_line = '%(mod_name)s.%(prefix)s%(func_name)s(%(f90_arg_names)s)' % dct
            if isinstance(node, Function):
                call_line = 'return %s' % call_line
            self.write(call_line)
            self.dedent()
            self.write()

    def visit_Type(self, node):
        logging.info('PythonWrapperGenerator visiting type %s' % node.name)
        cls_name = node.name.title()
        self.write('class %s(fortrantype.FortranDerivedType):' % cls_name)
        self.indent()
        self.write(self.format_doc_string(node))        
        self.generic_visit(node)

        for el in node.elements:
            dims = filter(lambda x: x.startswith('dimension'), el.attributes)
            if len(dims) == 0:  # proper scalar type (normal or derived)
                if el.type.startswith('type'):
                    self.write_dt_wrappers(node, el) 
                else:
                    self.write_scalar_wrappers(node, el) 
            elif el.type.startswith('type'):  # array of derived types
                self.write_dt_array_wrapper(node, el, dims)
            else:
                self.write_sc_array_wrapper(node, el, dims)
        self.dedent()

    def write_scalar_wrappers(self, node, el):
        dct = dict(el_name=el.name, mod_name=self.f90_mod_name,
                   prefix=self.prefix, type_name=node.name,
                   handle=isinstance(node, Type) and 'self._handle' or '')
        if isinstance(node, Type):
            dct['set_args'] = '%(handle)s, %(el_name)s' % dct
        else:
            dct['set_args'] = '%(el_name)s' % dct
            
        self.write('''@property
def %(el_name)s(self):''' % dct)
        self.indent()
        self.write(self.format_doc_string(el))
        self.write('return %(mod_name)s.%(prefix)s%(type_name)s__get__%(el_name)s(%(handle)s)' % dct)
        self.dedent()
        self.write()
        self.write('''@%(el_name)s.setter
def %(el_name)s(self, %(el_name)s):
    %(mod_name)s.%(prefix)s%(type_name)s__set__%(el_name)s(%(set_args)s)
''' % dct)
        self.write()

    def write_dt_wrappers(self, node, el):
        cls_name = strip_type(el.type).title()
        cls_mod_name = self.types[strip_type(el.type)].mod_name
        dct = dict(el_name=el.name, mod_name=self.f90_mod_name,
                   prefix=self.prefix, type_name=node.name,
                   cls_name=cls_name,
                   cls_mod_name=cls_mod_name,
                   handle=isinstance(node, Type) and 'self._handle' or '')
        if isinstance(node, Type):
            dct['set_args'] = '%(handle)s, %(el_name)s' % dct
        else:
            dct['set_args'] = '%(el_name)s' % dct
        self.write('''@property
def %(el_name)s(self):''' % dct)
        self.indent()
        self.write(self.format_doc_string(el))
        self.write('''%(el_name)s_handle = %(mod_name)s.%(prefix)s%(type_name)s__get__%(el_name)s(%(handle)s)
if tuple(%(el_name)s_handle) in self._objs:
    %(el_name)s = self._objs[tuple(%(el_name)s_handle)]
else:
    %(el_name)s = %(cls_mod_name)s.%(cls_name)s.from_handle(%(el_name)s_handle)
    self._objs[tuple(%(el_name)s_handle)] = %(el_name)s
return %(el_name)s''' % dct)
        self.dedent()
        self.write()
        self.write('''@%(el_name)s.setter
def %(el_name)s(self, %(el_name)s):
    %(el_name)s = %(el_name)s._handle
    %(mod_name)s.%(prefix)s%(type_name)s__set__%(el_name)s(%(set_args)s)
''' % dct)
        self.write()

    def write_sc_array_wrapper(self, node, el, dims):
        dct = dict(el_name=el.name, mod_name=self.f90_mod_name,
                   prefix=self.prefix, type_name=node.name,
                   doc=self.format_doc_string(el),
                   handle=isinstance(node, Type) and 'self._handle, ' or '[0]*_sizeof_fortran_t, ')
        self.write("""@property
def %(el_name)s(self):""" % dct)
        self.indent()
        self.write(self.format_doc_string(el))
        self.write("""if '%(el_name)s' in self._arrays:
    %(el_name)s = self._arrays['%(el_name)s']
else:
    %(el_name)s = arraydata.get_array(_sizeof_fortran_t,
                                      %(handle)s
                                      %(mod_name)s.%(prefix)s%(type_name)s__array__%(el_name)s)
    self._arrays['%(el_name)s'] = %(el_name)s
return %(el_name)s""" % dct)
        self.dedent()
        self.write()
        self.write("""@%(el_name)s.setter
def %(el_name)s(self, %(el_name)s):
    self.%(el_name)s[...] = %(el_name)s
""" % dct)
        self.write()

    def write_dt_array_wrapper(self, node, el, dims):
        pass
