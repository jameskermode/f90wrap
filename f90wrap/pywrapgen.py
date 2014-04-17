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
from f90wrap.fortran import *
from f90wrap.codegen import CodeGenerator
import numpy as np

class PythonWrapperGenerator(FortranVisitor, CodeGenerator):

    def __init__(self, prefix, mod_name, imports=None):
        CodeGenerator.__init__(self, indent=' ' * 4,
                               max_length=80,
                               continuation='\\')
        FortranVisitor.__init__(self)
        self.prefix = prefix
        self.mod_name = mod_name
        if imports is None:
            imports = [(mod_name, None),
                       ('functools', None),
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
        self.generic_visit(node)
        py_wrapper_file = open('%s.py' % node.name.lower(), 'w')
        py_wrapper_file.write(str(self))
        py_wrapper_file.close()

    def write_constructor(self, node):
        handle_arg = Argument(name='handle',
                              filename=node.filename,
                              doc='Opaque reference to existing derived type instance',
                              lineno=node.lineno,
                              attributes=['intent(in)'],
                              type='integer',
                              value=None)
        
        dct = dict(func_name=node.name,
                   prefix=self.prefix,
                   mod_name=self.mod_name,
                   py_arg_names=', '.join([arg.name for arg in node.arguments]),
                   f90_arg_names=', '.join(['%s=%s' % (arg.orig_name, arg.name) for arg in node.arguments])

        self.write("""@functools.wraps(%(mod_name)s.%(prefix)s%(func_name)s, assigned=['__doc__'])
def __init__(%(py_arg_names)s):""" % dct)
        self.indent()
        self.write('fortrantype.FortranDerivedType.__init__(self)')
        self.write('self._alloc = handle is None')
        self.write('if self._alloc:')
        self.indent()
        self.write('handle = %(mod_name)s.%(prefix)s%(func_name)s(%(f90_arg_names)s)' % dct)
        self.dedent()
        self.write('self._handle = handle')
        self.dedent()
        self.write()

    def write_destructor(self, node):
        dct = dict(func_name=node.name,
                   prefix=self.prefix,
                   mod_name=self.mod_name,
                   arg_names=', '.join([arg.name for arg in node.arguments]),
                   f90_arg_names=', '.join(['%s=%s' % (arg.orig_name, arg.name) for arg in node.arguments]))
        self.write("""@functools.wraps(%(mod_name)s.%(prefix)s%(func_name)s, assigned=['__doc__'])
def __del__(%(arg_names)s):""" % dct)
        self.indent()
        self.write('if self._alloc:')
        self.indent()
        self.write('%(mod_name)s.%(prefix)s%(func_name)s(%(arg_names)s)' % dct)
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
                       arg_names=', '.join([arg.name for arg in node.arguments]))
                       
            self.write("""@functools.wraps(%(mod_name)s.%(prefix)s%(func_name)s, assigned=['__doc__'])
def %(func_name)s(%(arg_names)s):""" % dct)
            self.indent()
            call_line = '%(mod_name)s.%(prefix)s%(func_name)s(%(arg_names)s)' % dct
            if isinstance(node, Function):
                call_line = 'return %s' % call_line
            self.write(call_line)
            self.dedent()
            self.write()

    def visit_Type(self, node):
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
                   prefix=self.prefix, type_name=node.name)
        self.write("""@property
def %(el_name)s(self):
    return %(mod_name)s.%(prefix)s%(type_name)s__get__%(el_name)s(self._handle)

@%(el_name)s.setter
def %(el_name)s(self, %(el_name)s):
    %(mod_name)s.%(prefix)s%(type_name)s__set__%(el_name)s(self._handle, %(el_name)s)
""" % dct)
        self.write()

    def write_dt_wrappers(self, node, el):
        dct = dict(el_name=el.name, mod_name=self.mod_name,
                   prefix=self.prefix, type_name=node.name,
                   cls_name=node.name.title())
        self.write("""@property
def %(el_name)s(self):
    %(el_name)s_handle = %(mod_name)s.%(prefix)s%(type_name)s__get__%(el_name)s(self._handle)
    if tuple(%(el_name)s_handle) in self._objs:
        %(el_name)s = self._objs[tuple(%(el_name)s_handle)]
    else:
        %(el_name)s = %(cls_name)s(handle=%(el_name)s_handle)
        self._objs[tuple(%(el_name)s_handle)] = %(el_name)s
    return %(el_name)s

@%(el_name)s.setter
def %(el_name)s(self, %(el_name)s):
    %(el_name)s = %(el_name)s._handle
    %(mod_name)s.%(prefix)s%(type_name)s__set__%(el_name)s(self._handle, %(el_name)s)
""" % dct)

    def write_sc_array_wrapper(self, node, el, dims):
        dct = dict(el_name=el.name, mod_name=self.mod_name,
                   prefix=self.prefix, type_name=node.name)
        self.write("""@property
def %(el_name)s(self):
   if '%(el_name)s' in self._arrays:
       %(el_name)s = self._arrays['%(el_name)s']
   else:
       %(el_name)s = arraydata.get_array(len(self._handle),
                                         self._handle,
                                         %(mod_name)s.%(prefix)s%(type_name)s__array__%(el_name)s)
       self._arrays['%(el_name)s'] = %(el_name)s
   return %(el_name)s

@%(el_name)s.setter
def %(el_name)s(self, %(el_name)s):
    self.%(el_name)s[...] = %(el_name)s

""" % dct)     

    def write_dt_array_wrapper(self, node, el, dims):
        pass
