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

    def __init__(self, prefix, mod_name, types, imports=None):
        CodeGenerator.__init__(self, indent=' ' * 4,
                               max_length=80,
                               continuation='\\')
        FortranVisitor.__init__(self)
        self.prefix = prefix
        self.mod_name = mod_name
        self.types = types
        if imports is None:
            imports = [(mod_name, None),
                       ('functools', None),
                       ('f90wrap.sizeof_fortran_t', 'sizeof_fortran_t'),
                       ('f90wrap.arraydata', 'arraydata'),
                       ('f90wrap.fortrantype', 'fortrantype')]
        self.imports = imports

    def visit_Module(self, node):
        self.code = []
        for (mod, alias) in self.imports:
            if alias is None:
                self.write('import %s' % mod)
            else:
                self.write('import %s as %s' % (mod, alias)) 
        self.write()
        self.write('_sizeof_fortran_t = sizeof_fortran_t.sizeof_fortran_t()')
        self.write()

        for el in node.elements:
            if el.type.startswith('type'):
                mod_name = self.types[el.type].mod_name
                cls_name = strip_type(el.type).title()
                if mod_name != node.name:
                    self.write('from %s import %s' % (mod_name, cls_name))
        self.write()        

        cls_name = node.name.title()
        self.write('class %s(fortrantype.FortranModule):' % cls_name)
        self.indent()
        if len(node.elements) == 0:
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
        self.dedent()
        self.write()
        self.write('fmod = %s()' % node.name.title())
        self.write()
        
        self.generic_visit(node)
        py_wrapper_file = open('%s.py' % node.name.lower(), 'w')
        py_wrapper_file.write(str(self))
        py_wrapper_file.close()
            

    def write_constructor(self, node):
        handle_arg = Argument(name='handle',
                              filename=node.filename,
                              doc='Opaque reference to existing derived type instance',
                              lineno=node.lineno,
                              attributes=['intent(in)', 'optional'],
                              type='integer')

        # special case for constructors: return value is 'self' argument,
        # plus we add an extra optional argument
        args = node.ret_val + node.arguments + [handle_arg]
        
        dct = dict(func_name=node.name,
                   prefix=self.prefix,
                   mod_name=self.mod_name,
                   py_arg_names=', '.join(['%s%s' % (arg.name,
                                                     'optional' in arg.attributes and '=None' or '')
                                                     for arg in args ]),
                   f90_arg_names=', '.join(['%s=%s' % (arg.orig_name, arg.value) for arg in node.arguments]))

        self.write("""@functools.wraps(%(mod_name)s.%(prefix)s%(func_name)s, assigned=['__doc__'])
def __init__(%(py_arg_names)s):""" % dct)
        self.indent()
        self.write('fortrantype.FortranDerivedType.__init__(self)')
        self.write('self._handle = %(mod_name)s.%(prefix)s%(func_name)s(%(f90_arg_names)s)' % dct)
        self.dedent()
        self.write()

    def write_destructor(self, node):
        dct = dict(func_name=node.name,
                   prefix=self.prefix,
                   mod_name=self.mod_name,
                   py_arg_names=', '.join(['%s%s' % (arg.name,
                                                     'optional' in arg.attributes and '=None' or '')
                                                     for arg in node.arguments]),
                   f90_arg_names=', '.join(['%s=%s' % (arg.orig_name, arg.value) for arg in node.arguments]))
        self.write("""@functools.wraps(%(mod_name)s.%(prefix)s%(func_name)s, assigned=['__doc__'])
def __del__(%(py_arg_names)s):""" % dct)
        self.indent()
        self.write('if self._alloc:')
        self.indent()
        self.write('%(mod_name)s.%(prefix)s%(func_name)s(%(f90_arg_names)s)' % dct)
        self.dedent()
        self.dedent()
        self.write()

    def visit_Procedure(self, node):
        if 'constructor' in node.attributes:
            self.write_constructor(node)
        elif 'destructor' in node.attributes:
            self.write_destructor(node)
        else:
            dct = dict(func_name=node.name,
                       prefix=self.prefix,
                       mod_name=self.mod_name,
                       py_arg_names=', '.join(['%s%s' % (arg.name,
                                                     arg.value is None and '=None' or '')
                                                     for arg in node.arguments ]),
                       f90_arg_names=', '.join(['%s=%s' % (arg.orig_name, arg.name) for arg in node.arguments]))
                       
            self.write("""@functools.wraps(%(mod_name)s.%(prefix)s%(func_name)s, assigned=['__doc__'])
def %(func_name)s(%(py_arg_names)s):""" % dct)
            self.indent()
            call_line = '%(mod_name)s.%(prefix)s%(func_name)s(%(f90_arg_names)s)' % dct
            if isinstance(node, Function):
                call_line = 'return %s' % call_line
            self.write(call_line)
            self.dedent()
            self.write()

    def visit_Type(self, node):
        for el in node.elements:
            if el.type.startswith('type'):
                mod_name = self.types[el.type].mod_name
                cls_name = strip_type(el.type).title()
                # FIXME check if type is defined in same module: no need to improt if so
                self.write('from %s import %s' % (mod_name, cls_name))
        self.write()

        cls_name = node.name.title()
        self.write('class %s(fortrantype.FortranDerivedType):' % cls_name)
        self.indent()
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
        dct = dict(el_name=el.name, mod_name=self.mod_name,
                   prefix=self.prefix, type_name=node.name,
                   handle=isinstance(node, Type) and 'self._handle' or '')
        if isinstance(node, Type):
            dct['set_args'] = '%(handle)s, %(el_name)s' % dct
        else:
            dct['set_args'] = '%(el_name)s' % dct
            
        self.write("""@property
def %(el_name)s(self):
    return %(mod_name)s.%(prefix)s%(type_name)s__get__%(el_name)s(%(handle)s)

@%(el_name)s.setter
def %(el_name)s(self, %(el_name)s):
    %(mod_name)s.%(prefix)s%(type_name)s__set__%(el_name)s(%(set_args)s)
""" % dct)
        self.write()

    def write_dt_wrappers(self, node, el):
        cls_name = strip_type(el.type).title()
        dct = dict(el_name=el.name, mod_name=self.mod_name,
                   prefix=self.prefix, type_name=node.name,
                   cls_name=cls_name,
                   handle=isinstance(node, Type) and 'self._handle' or '')
        if isinstance(node, Type):
            dct['set_args'] = '%(handle)s, %(el_name)s' % dct
        else:
            dct['set_args'] = '%(el_name)s' % dct
        self.write("""@property
def %(el_name)s(self):
    %(el_name)s_handle = %(mod_name)s.%(prefix)s%(type_name)s__get__%(el_name)s(%(handle)s)
    if tuple(%(el_name)s_handle) in self._objs:
        %(el_name)s = self._objs[tuple(%(el_name)s_handle)]
    else:
        %(el_name)s = %(cls_name)s.from_handle(%(el_name)s_handle)
        self._objs[tuple(%(el_name)s_handle)] = %(el_name)s
    return %(el_name)s

@%(el_name)s.setter
def %(el_name)s(self, %(el_name)s):
    %(el_name)s = %(el_name)s._handle
    %(mod_name)s.%(prefix)s%(type_name)s__set__%(el_name)s(%(set_args)s)

""" % dct)
        self.write()

    def write_sc_array_wrapper(self, node, el, dims):
        dct = dict(el_name=el.name, mod_name=self.mod_name,
                   prefix=self.prefix, type_name=node.name,
                   handle=isinstance(node, Type) and 'self._handle, ' or '[0]*_sizeof_fortran_t, ')
        self.write("""@property
def %(el_name)s(self):
   if '%(el_name)s' in self._arrays:
       %(el_name)s = self._arrays['%(el_name)s']
   else:
       %(el_name)s = arraydata.get_array(_sizeof_fortran_t,
                                         %(handle)s
                                         %(mod_name)s.%(prefix)s%(type_name)s__array__%(el_name)s)
       self._arrays['%(el_name)s'] = %(el_name)s
   return %(el_name)s

@%(el_name)s.setter
def %(el_name)s(self, %(el_name)s):
    self.%(el_name)s[...] = %(el_name)s
""" % dct)
        self.write()

    def write_dt_array_wrapper(self, node, el, dims):
        pass
