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

# numeric codes for Fortran types.
# Those with suffix _A are 1D arrays, _A2 2D are arrays
T_NONE = 0
T_INTEGER = 1
T_REAL = 2
T_COMPLEX = 3
T_LOGICAL = 4

T_INTEGER_A = 5
T_REAL_A = 6
T_COMPLEX_A = 7
T_LOGICAL_A = 8
T_CHAR = 9

T_CHAR_A = 10
T_DATA = 11
T_INTEGER_A2 = 12
T_REAL_A2 = 13

class F90WrapperGenerator(FortranVisitor, CodeGenerator):

    def __init__(self, prefix, sizeof_fortran_t, string_lengths):
        CodeGenerator.__init__(self, indent=' ' * 4,
                               max_length=80,
                               continuation='&')
        FortranVisitor.__init__(self)
        self.prefix = prefix
        self.sizeof_fortran_t = sizeof_fortran_t
        self.string_lengths = string_lengths

    def visit_Module(self, node):
        self.code = []
        self.generic_visit(node)
        f90_wrapper_file = open('%s%s.f90' % (self.prefix, node.name), 'w')
        f90_wrapper_file.write(str(self))
        f90_wrapper_file.close()

    def write_uses_lines(self, node):
        self.write('! BEGIN write_uses_lines')
        if hasattr(node, 'uses'):
            for (mod, only) in node.uses:
                if only is not None:
                    self.write('use %s, only: %s' % (mod, ' '.join(only)))
                else:
                    self.write('use %s' % mod)
        self.write('! END write_uses_lines')
        self.write()

    def write_type_lines(self, tname):
        """
        Write type definition for input type name
        """
        tname = _strip_type(tname)
        self.write("""type %(typename)s_ptr_type
    type(%(typename)s), pointer :: p => NULL()
end type %(typename)s_ptr_type""" % {'typename': tname})
        self.write()

    def write_arg_decl_lines(self, node):
        self.write('! BEGIN write_arg_decl_lines ')

        for arg in node.arguments:
            attributes = [attr for attr in arg.attributes if attr in ('optional', 'pointer', 'intent(in)',
                                                                       'intent(out)', 'intent(inout)') or
                                                                       attr.startswith('dimension') ]
            arg_dict = {'arg_type': arg.type,
                        'type_name': arg.type.startswith('type') and arg.type[5:-1] or None,
                        'arg_name': arg.name}  # self.prefix+arg.name}

            if arg.name in node.transfer_in or arg.name in node.transfer_out:
                self.write('type(%(type_name)s_ptr_type) :: %(arg_name)s_ptr' % arg_dict)
                arg_dict['arg_type'] = arg.wrapper_type
                attributes.append('dimension(%d)' % arg.wrapper_dim)

            arg_dict['arg_attribs'] = ', '.join(attributes)
            arg_dict['comma'] = len(attributes) != 0 and ', ' or ''

            self.write('%(arg_type)s%(comma)s%(arg_attribs)s :: %(arg_name)s' % arg_dict)
            if hasattr(arg, 'f2py_line'):
                self.write(arg.f2py_line)
        self.write('! END write_arg_decl_lines ')
        self.write()

    def write_transfer_in_lines(self, node):
        self.write('! BEGIN write_transfer_in_lines ')
        for arg in node.arguments:
            arg_dict = {'arg_name': arg.name,  # self.prefix+arg.name,
                        'arg_type': arg.type}
            if arg.name in node.transfer_in:
                if 'optional' in arg.attributes:
                    self.write("if (present(%(arg_name)s)) then" % arg_dict)
                    self.indent()

                self.write('%(arg_name)s_ptr = transfer(%(arg_name)s, %(arg_name)s_ptr)' % arg_dict)

                if 'optional' in arg.attributes:
                    self.dedent()
                    self.write('else')
                    self.indent()
                    self.write('%(arg_name)s_ptr%%p => null()' % arg_dict)
                    self.dedent()
                    self.write('end if')
        self.write('! END write_transfer_in_lines ')
        self.write()

    def write_init_lines(self, node):
        self.write('! BEGIN write_init_lines ')
        for alloc in node.allocate:
            self.write('allocate(%s_ptr%%p)' % alloc)  # (self.prefix, alloc))
        for arg in node.arguments:
            if not hasattr(arg, 'init_lines'):
                continue
            exe_optional, exe = arg.init_lines
            D = {'OLD_ARG':arg.name,
                 'ARG':arg.name,  # self.prefix+arg.name,
                 'PTR':arg.name + '_ptr%p'}
            if 'optional' in arg.attributes:
                self.write(exe_optional % D)
            else:
                self.write(exe % D)
        self.write('! END write_init_lines ')
        self.write()

    def write_call_lines(self, node):
        if 'skip_call' in node.attributes:
            return

        self.write('! BEGIN write_call_lines ')
        if hasattr(node, 'orig_node'):
            node = node.orig_node

        def dummy_arg_name(arg):
            return arg.name

        def actual_arg_name(arg):
            if arg.name in node.transfer_in or arg.name in node.transfer_out:
                return '%s_ptr%%p' % arg.name
            else:
                return arg.name

        arg_names = ['%s=%s' % (dummy_arg_name(arg), actual_arg_name(arg)) for arg in node.arguments
                     if 'intent(hide)' not in arg.attributes]
        if isinstance(node, Function):
            self.write('%(ret_val)s = %(func_name)s(%(arg_names)s)' %
                       {'ret_val': node.ret_val.name,
                        'func_name': node.name,
                        'arg_names': ', '.join(arg_names)})
        else:
            self.write('call %(sub_name)s(%(arg_names)s)' %
                       {'sub_name': node.name,
                        'arg_names': ', '.join(arg_names)})
        self.write('! END write_call_lines ')
        self.write()

    def write_transfer_out_lines(self, node):
        self.write('! BEGIN write_transfer_out_lines ')
        for arg in node.arguments:
            if arg.name in node.transfer_out:
                self.write('%(arg_name)s = transfer(%(arg_name)s_ptr, %(arg_name)s)' %
                           {'arg_name': arg.name})
        self.write('! END write_transfer_out_lines ')
        self.write()

    def write_finalise_lines(self, node):
        self.write('! BEGIN write_finalise_lines')
        for dealloc in node.deallocate:
            self.write('deallocate(%s_ptr%%p)' % dealloc)  # (self.prefix, dealloc))
        self.write('! END write_finalise_lines')
        self.write()

    def visit_Subroutine(self, node):
        print 'Visiting subroutine %s' % node.name

        self.write("subroutine %(sub_name)s(%(arg_names)s)" %
                   {'sub_name': self.prefix + node.name,
                    'arg_names': ', '.join([arg.name for arg in node.arguments])})
        self.indent()
        self.write()
        self.write_uses_lines(node)
        self.write("implicit none")
        self.write()
        self.write('! BEGIN write_type_lines')
        for tname in node.types:
            self.write_type_lines(tname)
        self.write("! END write_type_lines")
        self.write_arg_decl_lines(node)
        self.write_transfer_in_lines(node)
        self.write_init_lines(node)
        self.write_call_lines(node)
        self.write_transfer_out_lines(node)
        self.write_finalise_lines(node)
        self.dedent()
        self.write("end subroutine %(sub_name)s" % {'sub_name': self.prefix + node.name})
        self.write()
        return self.generic_visit(node)

    def visit_Type(self, node):
        print 'Visiting type %s' % node.name

        for el in node.elements:  # assuming Type has elements
            # Get the number of dimensions of the element (if any)
            dims = filter(lambda x: x.startswith('dimension'), el.attributes)
            # Skip this if the type is not do-able
            if 'pointer' in el.attributes and dims != []: continue
            if el.type.lower() == 'type(c_ptr)': continue

            if len(dims) == 0:  # proper scalar type (normal or derived)
                self.write_scalar_wrappers(node, el, self.sizeof_fortran_t)  # where to get sizeof_fortran_t from??
            elif el.type.startswith('type'):  # array of derived types
                self.write_dt_array_wrapper(node, el, dims, self.sizeof_fortran_t)  # where to get dims from?
            else:
                self.write_sc_array_wrapper(node, el, dims, self.sizeof_fortran_t)

        return self.generic_visit(node)

    def write_sc_array_wrapper(self, t, el, dims, sizeof_fortran_t):

        # The following maps the element type to a numeric code
        fortran_type_code = {
                             'd': T_REAL_A,
                             'i': T_INTEGER_A,
                             'S': T_CHAR_A,
                             'complex': T_COMPLEX_A
                             }

        numpy_type_map = {'real(8)': 'd',  # FIXME user-provided kinds should be included here
                          'real(dp)':'d',
                          'integer':'i',
                          'logical':'i',
                          'character*(*)':'S',
                          'complex(dp)':'complex',
                          'real(16)':'float128',
                          'real(qp)':'float128'}

        if el.type in numpy_type_map:
            typename = numpy_type_map[el.type]
        else:
            typename = el.type

        self.write('subroutine %s%s__array__%s(this, nd, dtype, dshape, dloc)' % (self.prefix, t.name, el.name))
        self.indent()
        self.write_uses_lines(t)
        self.write('implicit none')
        self.write_type_lines(t.name)
        self.write('integer, intent(in) :: this(%d)' % sizeof_fortran_t)
        self.write('type(%s_ptr_type) :: this_ptr' % t.name)
        self.write('integer, intent(out) :: nd')
        self.write('integer, intent(out) :: dtype')
        try:
            rank = dims[0].count(',') + 1
            if el.type.startswith('character'): rank += 1
        except ValueError:
            rank = 1
        self.write('integer, dimension(10), intent(out) :: dshape')
        self.write('integer*%d, intent(out) :: dloc' % np.dtype('O').itemsize)
        self.write()
        self.write('nd = %d' % rank)
        self.write('dtype = %s' % fortran_type_code[typename])
        self.write('this_ptr = transfer(this, this_ptr)')
        if 'allocatable' in el.attributes:
            self.write('if (allocated(this_ptr%%p%%%s)) then' % el.name)
            self.indent()
        if el.type.startswith('character'):
            first = ','.join(['1' for i in range(rank - 1)])
            self.write('dshape(1:%d) = (/len(this_ptr%%p%%%s(%s)), shape(this_ptr%%p%%%s)/)' % (rank, el.name, first, el.name))
        else:
            self.write('dshape(1:%d) = shape(this_ptr%%p%%%s)' % (rank, el.name))
        self.write('dloc = loc(this_ptr%%p%%%s)' % el.name)
        if 'allocatable' in el.attributes:
            self.dedent()
            self.write('else')
            self.indent()
            self.write('dloc = 0')
            self.dedent()
            self.write('end if')

        self.dedent()
        self.write('end subroutine %s%s__array__%s' % (self.prefix, t.name, el.name))
        self.write()

    def write_dt_array_wrapper(self, t, element, dims,
                            sizeof_fortran_t):
        """
        Write fortran get/set/len routine for an array
        
        Parameters
        ----------
        t : type
        element : element
        dims : dims
        sizeof_fortran_t : sizeof_fortran_t
        """
        if element.type.startswith('type') and len(dims) != 1:
            return

        self._write_array_getset_item(t, element, sizeof_fortran_t, 'get')
        self._write_array_setset_item(t, element, sizeof_fortran_t, 'set')
        self._write_array_len(t, element, sizeof_fortran_t)

    def write_scalar_wrappers(self, t, element, sizeof_fortran_t):

        self.write_scalar_wrapper(t, element, sizeof_fortran_t, "get")
        self.write_scalar_wrapper(t, element, sizeof_fortran_t, "set")

    def _write_array_getset_item(self, t, el, sizeof_fortran_t, getset):
        # getset and inout just change things simply from a get to a set routine.
        inout = "in"
        if getset == "get":
            inout = "out"

        self.write('subroutine %s%s__array_%sitem__%s(this, i, %s)' % (self.prefix, t.name,
                                                                       getset, el.name,
                                                                       el.name))
        self.indent()
        self.write()
        self.write_uses_lines(t)
        self.write('implicit none')
        self.write()
        self.write_type_lines(t.name)  # FIXME: not sure if this is right??!!
        self.write_type_lines(el.type)  # I'm passing strings!

        self.write('integer, intent(in) :: this(%d)' % sizeof_fortran_t)
        self.write('type(%s_ptr_type) :: this_ptr' % t.name)
        self.write('integer, intent(in) :: i')
        self.write('integer, intent(%s) :: %s(%d)' % (inout, el.name, sizeof_fortran_t))
        self.write('type(%s_ptr_type) :: %s_ptr' % (t.name, el.name))
        self.write()
        self.write('this_ptr = transfer(this, this_ptr)')

        if 'allocatable' in el.attributes:
            self.write('if (allocated(this_ptr%%p%%%s)) then' % el.name)
            self.indent()

        self.write('if (i < 1 .or. i > size(this_ptr%%p%%%s)) then' % el.name)
        self.indent()
        self.write('call system_abort("array index out of range")')
        self.dedent()
        self.write('else')
        self.indent()
        if getset == "get":
            self.write('%s_ptr%%p => this_ptr%%p%%%s(i)' % (el.name, el.name))
            self.write('%s = transfer(%s_ptr,%s)' % (el.name, el.name, el.name))
        else:
            self.write('%s_ptr = transfer(%s,%s_ptr)' % (el.name, el.name, el.name))
            self.write('this_ptr%%p%%%s(i) = %s_ptr%%p' % (el.name, el.name))

        self.dedent()
        self.write('endif')

        if 'allocatable' in el.attributes:
            self.dedent()
            self.write('else')
            self.indent()
            self.write('call system_abort("derived type array not allocated")')
            self.dedent()
            self.write('end if')

        self.dedent()
        self.write('end subroutine %s%s__array_%sitem__%s' % (self.prefix, t.name,
                                                              getset, el.name))
        self.write()


    def _write_array_len(self, t, el, sizeof_fortran_t):
        self.write('subroutine %s%s__array_len__%s(this, n)' % (self.prefix, t.name, el.name))
        self.indent()
        self.write()
        self.write_uses_lines(t)
        self.write('implicit none')
        self.write()
        self.write_type_lines(t.name)
        self.write_type_lines(el.type.name)
        self.write('integer, intent(in) :: this(%d)' % sizeof_fortran_t)
        self.write('integer, intent(out) :: n')
        self.write('type(%s_ptr_type) :: this_ptr' % t.name)
        self.write()
        self.write('this_ptr = transfer(this, this_ptr)')

        if 'allocatable' in el.attributes:
            self.write('if (allocated(this_ptr%%p%%%s)) then' % el.name)
            self.indent()

        self.write('n = size(this_ptr%%p%%%s)' % el.name)

        if 'allocatable' in el.attributes:
            self.dedent()
            self.write('else')
            self.indent()
            self.write('n = 0')
            self.dedent()
            self.write('end if')

        self.dedent()
        self.write('end subroutine %s%s__array_len__%s' % (self.prefix, t.name, el.name))
        self.write()


    def write_scalar_wrapper(self, t, el, sizeof_fortran_t, getset):
        """
        t : a type object from the big tree thing
        element : an element within this type.
        getset : either 'get' or 'set'
        """
        # getset and inout just change things simply from a get to a set routine.
        inout = "in"
        if getset == "get":
            inout = "out"

        self.write('subroutine %s%s__%s__%s(this, %s)' % (self.prefix, t.name,
                                                             getset, el.name, el.name))
        self.indent()
        self.write_uses_lines(t)
        self.write('implicit none')
        self.write_type_lines(t.name)
        if el.type.startswith('type'):
            self.write_type_lines(el.type)

        self.write('integer, intent(in)   :: this(%d)' % sizeof_fortran_t)
        self.write('type(%s_ptr_type) :: this_ptr' % t.name)

        if el.type.startswith('type'):
            # For derived types elements, treat as opaque reference
            self.write('integer, intent(%s) :: %s(%d)' % (inout, el.name, sizeof_fortran_t))

            self.write('type(%s_ptr_type) :: %s_ptr' % (_strip_type(el.type), el.name))
            self.write()
            self.write('this_ptr = transfer(this, this_ptr)')
            if getset == "get":
                self.write('%s_ptr%%p => this_ptr%%p%%%s' % (el.name, el.name))
                self.write('%s = transfer(%s_ptr,%s)' % (el.name, el.name, el.name))
            else:
                self.write('%s_ptr = transfer(%s,%s_ptr)' % (el.name,
                                                                      el.name,
                                                                      el.name))
                self.write('this_ptr%%p%%%s = %s_ptr%%p' % (el.name, el.name))
        else:
            # Return/set by value
            if 'pointer' in el.attributes:
                el.attributes.remove('pointer')

            if el.attributes != []:
                self.write('%s, %s, intent(%s) :: %s' % (el.type,
                                                         ','.join(el.attributes),
                                                         inout, el.name))
            else:
                self.write('%s, intent(%s) :: %s' % (el.type, inout, el.name))
            self.write()
            self.write('this_ptr = transfer(this, this_ptr)')
            if getset == "get":
                self.write('%s = this_ptr%%p%%%s' % (el.name, el.name))
            else:
                self.write('this_ptr%%p%%%s = %s' % (el.name, el.name))
        self.dedent()
        self.write('end subroutine %s%s__%s__%s' % (self.prefix, t.name, getset,
                                                    el.name))
        self.write()


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

    def _fmt_arg(self, node, arg, replace_types=True, include_values=False):
        arg_name = arg.name
        if arg.name == 'this':
            arg_name = 'self'
        if replace_types and arg.type.startswith('type'):
            arg_name = '%s._handle' % arg_name
        arg_str = arg_name
        if include_values and arg.value != '':
            arg_str += '=%r' % arg.value
        return arg_str

    def _skip_arg(self, node, arg):
        if 'intent(out)' in arg.attributes:
            return True
        return False

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

        f90_arguments = [arg for arg in node.arguments if not self._skip_arg(node, arg)]
        py_arguments = node.arguments + [handle_arg]

        dct = dict(func_name=node.name,
                   prefix=self.prefix,
                   mod_name=self.mod_name,
                   py_arg_names=', '.join([self._fmt_arg(node, arg, False, True) for arg in py_arguments]),
                   f90_arg_names=', '.join([self._fmt_arg(node, arg) for arg in f90_arguments]))

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
                   arg_names=', '.join([self._fmt_arg(node, arg, False, False) for arg in node.arguments if not
                                        self._skip_arg(node, arg)]))
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
                       arg_names=', '.join([self._fmt_arg(node, arg) for arg in node.arguments]))
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
            # Get the number of dimensions of the element (if any)
            dims = filter(lambda x: x.startswith('dimension'), el.attributes)
            # Skip this if the type is not do-able
            if 'pointer' in el.attributes and dims != []: continue
            if el.type.lower() == 'type(c_ptr)': continue

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


class UnwrappablesRemover(FortranTransformer):

    def __init__(self, callbacks, types, constructors, destructors):
        self.callbacks = callbacks
        self.types = types
        self.constructors = constructors
        self.destructors = destructors

    def visit_Procedure(self, node):
        # special case: keep all constructors and destructors, although
        # they may have pointer arguments
        for suff in self.constructors + self.destructors:
            if node.name.endswith(suff):
                return node

        args = node.arguments[:]
        if isinstance(node, Function):
            args.append(node.ret_val)

        for arg in args:
            # only callback functions in self.callbacks
            if 'callback' in arg.attributes:
                if node.name not in self.callbacks:
                    logging.debug('removing callback routine %s' % node.name)
                    return None
                else:
                    continue

            if 'optional' in arg.attributes:
                # we can remove the argument instead of the whole routine
                return self.generic_visit(node)
            else:
                # no allocatables or pointers
                if 'allocatable' in arg.attributes or 'pointer' in arg.attributes:
                    logging.debug('removing routine %s due to allocatable/pointer arguments' % node.name)
                    return None

                # no complex scalars (arrays are OK)
                dims = [attrib for attrib in arg.attributes if attrib.startswith('dimension')]
                if arg.type.startswith('complex') and len(dims) == 0:
                    logging.debug('removing routine %s due to complex scalar arguments' % node.name)
                    return None

                # no derived types apart from those in self.types
                if arg.type.startswith('type') and arg.type not in self.types:
                    logging.debug('removing routine %s due to unsupported derived type %s' % (node.name, arg.type))
                    return None

        return self.generic_visit(node)

    def visit_Argument(self, node):
        if not hasattr(node, 'attributes'):
            return node

        if not 'optional' in node.attributes:
            return node

        # remove optional allocatable/pointer arguments
        if 'allocatable' in node.attributes or 'pointer' in node.attributes:
            logging.debug('removing optional argument %s due to allocatable/pointer attributes' % node.name)
            return None

        # remove optional complex scalar arguments
        dims = [attrib for attrib in node.attributes if attrib.startswith('dimension')]
        if node.type.startswith('complex') and len(dims) == 0:
            logging.debug('removing optional argument %s as it is a complex scalar' % node.name)
            return None

        # remove optional derived types not in self.types
        if node.type.startswith('type') and node.type not in self.types:
            logging.debug('removing optional argument %s due to unsupported derived type %s' % (node.name, node.type))
            return None

        return node

    def visit_Type(self, node):
        if node.name not in self.types:
            logging.debug('removing type %s' % node.name)
            return None
        else:
            return node



def fix_subroutine_uses_clauses(tree, types, kinds):
    """Walk over all nodes in tree, updating subroutine uses
       clauses to include the parent module and all necessary modules
       from types and kinds."""

    for mod, sub, arguments in walk_procedures(tree):

        sub.uses = set()
        sub.uses.add((mod.name, None))

        for arg in arguments:
            if arg.type.startswith('type') and arg.type in types:
                sub.uses.add((types[arg.type].mod_name, None))

        for (mod, only) in kinds:
            if mod not in sub.uses:
                sub.uses.add((mod, only))

    return tree

def set_intent(attributes, intent):
    """Remove any current "intent" from attributes and replace with intent given"""
    attributes = [attr for attr in attributes if not attr.startswith('intent')]
    attributes.append(intent)
    return attributes

def convert_derived_type_arguments(tree, init_lines, sizeof_fortran_t):
    for mod, sub, arguments in walk_procedures(tree):
        sub.types = set()
        sub.transfer_in = []
        sub.transfer_out = []
        sub.allocate = []
        sub.deallocate = []

        if 'constructor' in sub.attributes:
            sub.arguments[0].attributes = set_intent(sub.arguments[0].attributes, 'intent(out)')

        if 'destructor' in sub.attributes:
            logging.debug('deallocating arg "%s" in %s' % (sub.arguments[0].name, sub.name))
            sub.deallocate.append(sub.arguments[0].name)

        for arg in arguments:
            if not arg.type.startswith('type'):
                continue

            # save original Fortran intent since we'll be overwriting it
            # with intent of the opaque pointer
            arg.attributes = arg.attributes + ['fortran_' + attr for attr in
                               arg.attributes if attr.startswith('intent')]

            typename = _strip_type(arg.type)
            arg.wrapper_type = 'integer'
            arg.wrapper_dim = sizeof_fortran_t
            sub.types.add(typename)

            if typename in init_lines:
                use, (exe, exe_optional) = init_lines[typename]
                if use is not None:
                    sub.uses.add((use, None))
                arg.init_lines = (exe_optional, exe)

            if 'intent(out)' in arg.attributes:
                arg.attributes = set_intent(arg.attributes, 'intent(out)')
                sub.transfer_out.append(arg.name)
                if 'pointer' not in arg.attributes:
                    logging.debug('allocating arg "%s" in %s' % (arg.name, sub.name))
                    sub.allocate.append(arg.name)
            else:
                arg.attributes = set_intent(arg.attributes, 'intent(in)')
                sub.transfer_in.append(arg.name)

    return tree


class StringLengthConverter(FortranVisitor):
    """Convert lengths of all character strings to standard format

    Looks in all Procedure arguments and Type elements.
    Changes from '(len=*)' or '(*)' syntax to *(*) syntax.
    """

    def __init__(self, string_lengths, default_string_length):
        self.string_lengths = string_lengths
        self.default_string_length = default_string_length

    def visit_Declaration(self, node):
        if not node.type.startswith('character'):
            return

        try:
            lind = node.type.index('(')
            rind = node.type.rindex(')')
            typ = node.type[:lind] + '*' + node.type[lind:rind + 1].replace('len=', '')
            string_length = typ[11:-1]

            # Try to get length of string arguments
            if not string_length == '*' and not all([x in '0123456789' for x in string_length]):
                string_length = self.string_lengths.get(string_length, self.default_string_length)

            # Default string length for intent(out) strings
            if string_length == '*' and 'intent(out)' in node.attributes:
                string_length = 'character*(%s)' % self.default_string_length

        except ValueError:
            string_length = 1

        node.type = 'character*(%s)' % str(string_length)

class ArrayDimensionConverter(FortranVisitor):
    """
    Transform unspecified dimensions into additional dummy arguments

    e.g. the following code

        subroutine foo(a)
          integer a(:)
        end subroutine foo

    becomes:

        subroutine foo(a, n0)
          integer a(n0)
          integer n0
          !f2py intent(hide), depend(a) :: n0 = shape(a,0)
        end subroutine foo
    """

    valid_dim_re = re.compile(r'^(([-0-9.e]+)|(size\([_a-zA-Z0-9\+\-\*\/]*\))|(len\(.*\)))$')

    @staticmethod
    def split_dimensions(dim):
        """Given a string like "dimension(a,b,c)" return the list of dimensions ['a','b','c']."""
        dim = dim[10:-1]  # remove "dimension(" and ")"
        br = 0
        d = 1
        ds = ['']
        for c in dim:
            if c != ',': ds[-1] += c
            if   c == '(': br += 1
            elif c == ')': br -= 1
            elif c == ',':
                if br == 0: ds.append('')
                else: ds[-1] += ','
        return ds

    def visit_Procedure(self, node):

        n_dummy = 0
        for arg in node.arguments:
            dims = [attr for attr in arg.attributes if attr.startswith('dimension(') ]
            if dims == []:
                continue
            if len(dims) != 1:
                raise ValueError('more than one dimension attribute found for arg %s in sub %s' %
                                 (arg.name, sub.name))

            ds = ArrayDimensionConverter.split_dimensions(dims[0])

            new_dummy_args = []
            new_ds = []
            for i, d in enumerate(ds):
                if ArrayDimensionConverter.valid_dim_re.match(d):
                    if d.startswith('len'):
                        arg.f2py_line = ('!f2py %s %s, dimension(%s) :: %s' % \
                                         (arg.type,
                                           ','.join([attr for attr in arg.attributes if not attr.startswith('dimension')]),
                                           d.replace('len', 'slen'), arg.name))
                    continue
                dummy_arg = Argument(name='n%d' % n_dummy, type='integer', attributes=['intent(hide)'])

                if 'intent(out)' not in arg.attributes:
                    dummy_arg.f2py_line = ('!f2py intent(hide), depend(%s) :: %s = shape(%s,%d)' %
                                           (arg.name, dummy_arg.name, arg.name, i))
                new_dummy_args.append(dummy_arg)
                new_ds.append(dummy_arg.name)
                n_dummy += 1

            if new_dummy_args != []:
                logging.debug('adding dummy arguments %r to %s' % (new_dummy_args, node.name))
                arg.attributes = ([attr for attr in arg.attributes if not attr.startswith('dimension(')] +
                                  ['dimension(%s)' % ','.join(new_ds)])
                node.arguments.extend(new_dummy_args)


class MethodFinder(FortranTransformer):

    def __init__(self, types, constructor_names, destructor_names, short_names):
        self.types = types
        self.constructor_names = constructor_names
        self.destructor_names = destructor_names
        self.short_names = short_names

    def visit_Interface(self, node):
        new_procs = []
        for proc in node.procedures:
            if isinstance(proc, Procedure):
                new_proc = self.visit_Procedure(proc, interface=node)
                if new_proc is not None:
                    new_procs.append(new_proc)
            else:
                new_procs.append(proc)

        if new_procs == []:
            # interface is now empty: all routines have been moved into Interfaces inside types
            return None
        else:
            # some procedures remain so we need to keep the Interface around
            node.procedures = new_procs
            return node

    def visit_Procedure(self, node, interface=None):
        if (len(node.arguments) == 0 or
             (node.arguments[0] > 0 and
              node.arguments[0].type not in self.types)):
            # procedure is not a method, so leave it alone
            return node

        node.attributes.append('method')
        typ = self.types[node.arguments[0].type]

        # remove prefix from subroutine name to get method name
        node.method_name = node.name
        prefices = [typ.name + '_']
        if typ.name in self.short_names:
            prefices.append(self.short_names[typ.name] + '_')
        for prefix in prefices:
            if node.name.startswith(prefix):
                node.method_name = node.name[len(prefix):]

        # label constructors and destructors
        if node.method_name in self.constructor_names:
            node.attributes.append('constructor')
        elif node.method_name in self.destructor_names:
            node.attributes.append('destructor')

        if interface is None:
            # just a regular method - move into typ.procedures
            typ.procedures.append(node)
            logging.debug('added method %s to type %s' %
                          (node.method_name, typ.name))
        else:
            # this method was originally inside an interface,
            # so we need to replicate Interface inside the Type
            for intf in typ.interfaces:
                if intf.name == interface.name:
                    intf.procedures.append(node)
                    logging.debug('added method %s to interface %s in type %s' %
                                  (node.method_name, intf.name, typ.name))
                    break
            else:
                intf = Interface(interface.name,
                                 interface.filename,
                                 interface.doc,
                                 interface.lineno,
                                 [node])
                typ.interfaces.append(intf)
                logging.debug('added method %s to interface %s in type %s' %
                              (node.method_name, intf.name, typ.name))

        # remove method from parent since we've added it to Type
        return None

def collapse_single_interfaces(tree):
    """Collapse interfaces which contain only a single procedure."""

    class _InterfaceCollapser(FortranTransformer):
        """Replace interfaces with only one procedure by that procedure"""
        def visit_Interface(self, node):
            if len(node.procedures) == 1:
                proc = node.procedures[0]
                proc.doc = node.doc + proc.doc
                logging.debug('collapsing single-component interface %s' % proc.name)
                return proc
            else:
                return node

    class _ProcedureRelocator(FortranTransformer):
        """Filter interfaces and procedures into correct lists"""
        def visit_Type(self, node):
            logging.debug('visiting %r' % node)
            interfaces = []
            procedures = []
            for child in iter_child_nodes(node):
                if isinstance(child, Interface):
                    interfaces.append(child)
                elif isinstance(child, Procedure):
                    procedures.append(child)
                else:
                    # other child nodes should be left where they are
                    pass

            node.interfaces = interfaces
            node.procedures = procedures
            return self.generic_visit(node)

        visit_Module = visit_Type

    tree = _InterfaceCollapser().visit(tree)
    tree = _ProcedureRelocator().visit(tree)
    return tree

def add_missing_constructors(tree):
    for node in walk(tree):
        if not isinstance(node, Type):
            continue
        for child in iter_child_nodes(node):
            if isinstance(child, Procedure):
                if 'constructor' in child.attributes:
                    print 'found constructor', child.name
                    break
        else:
            print 'adding missing constructor for %s' % node.name
            node.procedures.append(Subroutine('%s_initialise' % node.name,
                                              node.filename,
                                              'Automatically generated constructor for %s' % node.name,
                                              node.lineno,
                                              [Argument(name='this',
                                                        filename=node.filename,
                                                        doc='Object to be constructed',
                                                        lineno=node.lineno,
                                                        attributes=['intent(out)'],
                                                        type='type(%s)' % node.name)],
                                              node.uses,
                                              ['constructor', 'skip_call']))
    return tree


def add_missing_destructors(tree):
    for node in walk(tree):
        if not isinstance(node, Type):
            continue
        for child in iter_child_nodes(node):
            if isinstance(child, Procedure):
                if 'destructor' in child.attributes:
                    print 'found destructor', child.name
                    break
        else:
            print 'adding missing destructor for %s' % node.name
            node.procedures.append(Subroutine('%s_finalise' % node.name,
                                              node.filename,
                                              'Automatically generated destructor for %s' % node.name,
                                              node.lineno,
                                              [Argument(name='this',
                                                        filename=node.filename,
                                                        doc='Object to be destructed',
                                                        lineno=node.lineno,
                                                        attributes=['intent(inout)'],
                                                        type='type(%s)' % node.name)],
                                              node.uses,
                                              ['destructor', 'skip_call']))
    return tree


class FunctionToSubroutineConverter(FortranVisitor):
    """Convert all functions to subroutines, with return value as an
       intent(out) argument after the last non-optional argument"""

    def visit_Function(self, node):

        # insert ret_val after last non-optional argument
        arguments = node.arguments[:]
        i = 0
        for i, arg in enumerate(arguments):
            if 'optional' in arg.attributes:
                break
        arguments.insert(i, node.ret_val)
        arguments[i].name = 'ret_' + arguments[i].name
        arguments[i].attributes.append('intent(out)')

        new_node = Subroutine(node.name,
                              node.filename,
                              node.doc,
                              node.lineno,
                              arguments,
                              node.uses,
                              node.attributes)
        new_node.orig_node = node  # keep a reference to the original node
        return new_node

class OnlyAndSkip(FortranTransformer):
    """
    This class does the job of removing nodes from the tree 
    which are not necessary to write wrappers for (given user-supplied
    values for only and skip). 
    
    Currently it takes a list of subroutines and a list of types to write
    wrappers for. If empty, it does all of them. 
    """
    def __init__(self, kept_subs, kept_types):
        self.kept_subs = kept_subs
        self.kept_types = kept_types

    def visit_Procedure(self, node):
        if len(self.kept_subs) > 0:
            if node not in self.kept_subs:
                return None
        return node

    def visit_Type(self, node):
        if len(self.kept_types) > 0:
            if node not in self.kept_types:
                return None
        return node

def transform_to_f90_wrapper(tree, types, kinds, callbacks, constructors,
                              destructors, short_names, init_lines,
                              string_lengths, default_string_length,
                              sizeof_fortran_t, only_subs, only_types):

    """
    Apply a number of rules to *tree* to make it suitable for passing to
    a F90WrapperGenerator's visit() method. Transformations performed are:
 
     * Removal of procedures and types not provided by the user
     * Removal of private symbols
     * Removal of unwrappable routines and optional arguments
     * Addition of missing constructor and destructor wrappers
     * Conversion of all functions to subroutines
     * Update of subroutine uses clauses
     * Conversion of derived type arguments to opaque integer arrays
       via Fortran transfer() intrinsic.
     * ...
    """
    tree = OnlyAndSkip(only_subs, only_types).visit(tree)
    tree = remove_private_symbols(tree)
    tree = UnwrappablesRemover(callbacks, types, constructors, destructors).visit(tree)
    tree = MethodFinder(types, constructors, destructors, short_names).visit(tree)
    tree = collapse_single_interfaces(tree)

    FunctionToSubroutineConverter().visit(tree)

    tree = fix_subroutine_uses_clauses(tree, types, kinds)
    tree = convert_derived_type_arguments(tree, init_lines, sizeof_fortran_t)
    StringLengthConverter(string_lengths, default_string_length).visit(tree)
    ArrayDimensionConverter().visit(tree)
    return tree


def transform_to_py_wrapper(tree, types, kinds, callbacks, constructors,
                            destructors, short_names, init_lines):

    tree = remove_private_symbols(tree)
    tree = UnwrappablesRemover(callbacks, types, constructors, destructors).visit(tree)
    tree = MethodFinder(types, constructors, destructors, short_names).visit(tree)
    tree = collapse_single_interfaces(tree)
    tree = add_missing_constructors(tree)
    tree = add_missing_destructors(tree)
    return tree


def _strip_type(t):
    if t.startswith('type('):
        t = t[t.index('(') + 1:t.index(')')]
    return t.lower()


def find_referenced_types(mods, tree):
    """
    Given a set of modules in a parse tree, find any types either defined in
    or referenced by the module, recursively.
    
    Parameters
    ----------
    mods : initial modules to search, must be included in the tree.
    
    tree : the full fortran parse tree from which the mods have been taken.
    
    Returns
    -------
    kept_types : set of Type() objects which are referenced or defined in the 
                 modules given, or recursively referenced by those types. 
    """

    # Get used types now
    kept_types = set()
    for mod in mods:
        for t in mod.types:
            kept_types.add(t)

        for el in mod.elements:
            if el.type.startswith('type'):
                for mod2 in walk_modules(tree):
                    for mt in mod2.types:
                        if mt.name in el.type:
                            kept_types.add(mt)

    # kept_types is now all types defined/referenced directly in kept_mods. But we also
    # need those referenced by them.
    new_set = copy.copy(kept_types)
    while new_set != set():
        temp_set = list(new_set)
        for t in temp_set:
            for el in t.elements:
                if el.type.startswith('type'):  # a referenced type, need to find def
                    for mod2 in walk_modules(tree):
                        for mt in mod2.types:
                            if mt.name in el.type:
                                new_set.add(mt)
        # take out all the original types from new_set
        new_set -= kept_types
        # update the kept_types with new ones
        kept_types |= new_set

    return kept_types