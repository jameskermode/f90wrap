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
from f90wrap.fortran import (Fortran, Root, Program, Module, Procedure, Subroutine, Function, Prototype,
                             Declaration, Element, Argument, Type, Interface, FortranVisitor, strip_type)
from f90wrap.codegen import CodeGenerator
import numpy as np

# numeric codes for Fortran types.
# Those with suffix _A are 1D arrays, _A2 are 2D arrays
T_NONE        =  0
T_INTEGER     =  1
T_REAL        =  2
T_COMPLEX     =  3
T_LOGICAL     =  4

T_INTEGER_A   =  5
T_REAL_A      =  6
T_COMPLEX_A   =  7
T_LOGICAL_A   =  8
T_CHAR        =  9

T_CHAR_A      =  10
T_DATA        =  11
T_INTEGER_A2  =  12
T_REAL_A2     =  13

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

        for el in node.elements:
            dims = filter(lambda x: x.startswith('dimension'), el.attributes)
            if len(dims) == 0:  # proper scalar type (normal or derived)
                self.write_scalar_wrappers(node, el, self.sizeof_fortran_t)  # where to get sizeof_fortran_t from??
            elif el.type.startswith('type'):  # array of derived types
                self.write_dt_array_wrapper(node, el, dims, self.sizeof_fortran_t)  # where to get dims from?
            else:
                self.write_sc_array_wrapper(node, el, dims, self.sizeof_fortran_t)
        
        if len(self.code) > 0:
            f90_wrapper_file = open('%s%s.f90' % (self.prefix, node.name), 'w')
            f90_wrapper_file.write(str(self))
            f90_wrapper_file.close()

    def write_uses_lines(self, node):
        if hasattr(node, 'uses'):
            for uses in node.uses:
                if isinstance(uses, tuple):
                    mod, only = uses
                else:
                    mod, only = uses, None
                if only is not None:
                    self.write('use %s, only: %s' % (mod, ' '.join(only)))
                else:
                    self.write('use %s' % mod)

    def write_type_lines(self, tname):
        """
        Write type definition for input type name
        """
        tname = strip_type(tname)
        self.write("""type %(typename)s_ptr_type
    type(%(typename)s), pointer :: p => NULL()
end type %(typename)s_ptr_type""" % {'typename': tname})

    def write_arg_decl_lines(self, node):
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

    def write_transfer_in_lines(self, node):
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

    def write_init_lines(self, node):
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

    def write_call_lines(self, node):
        if 'skip_call' in node.attributes:
            return

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

    def write_transfer_out_lines(self, node):
        for arg in node.arguments:
            if arg.name in node.transfer_out:
                self.write('%(arg_name)s = transfer(%(arg_name)s_ptr, %(arg_name)s)' %
                           {'arg_name': arg.name})

    def write_finalise_lines(self, node):
        for dealloc in node.deallocate:
            self.write('deallocate(%s_ptr%%p)' % dealloc)  # (self.prefix, dealloc))

    def visit_Subroutine(self, node):
        self.write("subroutine %(sub_name)s(%(arg_names)s)" %
                   {'sub_name': self.prefix + node.name,
                    'arg_names': ', '.join([arg.name for arg in node.arguments])})
        self.indent()
        self.write_uses_lines(node)
        self.write("implicit none")
        self.write()
        for tname in node.types:
            self.write_type_lines(tname)
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
        logging.info('Visiting type %s' % node.name)

        for el in node.elements:
            dims = filter(lambda x: x.startswith('dimension'), el.attributes)
            if len(dims) == 0:  # proper scalar type (normal or derived)
                self.write_scalar_wrappers(node, el, self.sizeof_fortran_t)  # where to get sizeof_fortran_t from??
            elif el.type.startswith('type'):  # array of derived types
                self.write_dt_array_wrapper(node, el, dims, self.sizeof_fortran_t)  # where to get dims from?
            else:
                self.write_sc_array_wrapper(node, el, dims, self.sizeof_fortran_t)

        return self.generic_visit(node)

    def write_sc_array_wrapper(self, t, el, dims, sizeof_fortran_t):

        # The following maps the element type to a numeric code
        fortran_type_code = {'d': T_REAL_A,
                             'i': T_INTEGER_A,
                             'S': T_CHAR_A,
                             'complex': T_COMPLEX_A}

        numpy_type_map = {'real(8)': 'd',  # FIXME user-provided kinds should be included here
                          'real(dp)':'d',
                          'real(dl)':'d',
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

        if isinstance(t, Type):
            this = 'this, '
        else:
            this = 'dummy_this, '

        self.write('subroutine %s%s__array__%s(%snd, dtype, dshape, dloc)' % (self.prefix, t.name, el.name, this))
        self.indent()
        self.write_uses_lines(t)
        if isinstance(t, Module):
            use_only = []
            use_only.append('%s_%s => %s' % (t.name, el.name, el.name))
            use_only = ', '.join(use_only)
            self.write('use %s, only: ' % t.name + use_only)        
        self.write('implicit none')
        if isinstance(t, Type):
            self.write_type_lines(t.name)
            self.write('integer, intent(in) :: this(%d)' % sizeof_fortran_t)
            self.write('type(%s_ptr_type) :: this_ptr' % t.name)
        else:
            self.write('integer, intent(in) :: dummy_this(%d)' % sizeof_fortran_t)
            
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
        if isinstance(t, Type):
            self.write('this_ptr = transfer(this, this_ptr)')
            array_name = 'this_ptr%%p%%%s' % el.name
        else:
            array_name = '%s_%s' % (t.name, el.name)
            
        if 'allocatable' in el.attributes:
            self.write('if (allocated(%s)) then' % array_name)
            self.indent()
        if el.type.startswith('character'):
            first = ','.join(['1' for i in range(rank - 1)])
            self.write('dshape(1:%d) = (/len(%s(%s)), shape(%s)/)' % (rank, array_name, first, array_name))
        else:
            self.write('dshape(1:%d) = shape(%s)' % (rank, array_name))
        self.write('dloc = loc(%s)' % array_name)
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
        self._write_array_getset_item(t, element, sizeof_fortran_t, 'set')
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
        self.write_type_lines(el.type.name)  # I'm passing strings!

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
        t : a Type or Module object from the parse tree
        element : an element within this type or module
        getset : either 'get' or 'set'
        """
        # getset and inout just change things simply from a get to a set routine.
        inout = "in"
        if getset == "get":
            inout = "out"

        if isinstance(t, Type):
            this = 'this, '
        elif isinstance(t, Module):
            this = ''
        else:
            raise ValueError("Don't know how to write scalar wrappers for %s type %s" (t, type(t)))
                        
        self.write('subroutine %s%s__%s__%s(%s%s)' % (self.prefix, t.name,
                                                    getset, el.name, this, el.name))
        self.indent()
        self.write_uses_lines(t)
        if isinstance(t, Module):
            use_only = []
            use_only.append('%s_%s => %s' % (t.name, el.name, el.name))
            use_only = ', '.join(use_only)
            self.write('use %s, only: ' % t.name + use_only)
            
        self.write('implicit none')
        if isinstance(t, Type):
            self.write_type_lines(t.name)
            
        if el.type.startswith('type'):
            self.write_type_lines(el.type)

        if isinstance(t, Type):
            self.write('integer, intent(in)   :: this(%d)' % sizeof_fortran_t)
            self.write('type(%s_ptr_type) :: this_ptr' % t.name)

        if el.type.startswith('type'):
            # For derived types elements, treat as opaque reference
            self.write('integer, intent(%s) :: %s(%d)' % (inout, el.name, sizeof_fortran_t))

            self.write('type(%s_ptr_type) :: %s_ptr' % (strip_type(el.type), el.name))
            self.write()
            if isinstance(t, Type):
                self.write('this_ptr = transfer(this, this_ptr)')
            if getset == "get":
                if isinstance(t, Type):
                    self.write('%s_ptr%%p => this_ptr%%p%%%s' % (el.name, el.name))
                else:
                    self.write('%s_ptr%%p => %s_%s' % (el.name, t.name, el.name))
                self.write('%s = transfer(%s_ptr,%s)' % (el.name, el.name, el.name))
            else:
                self.write('%s_ptr = transfer(%s,%s_ptr)' % (el.name,
                                                             el.name,
                                                             el.name))
                if isinstance(t, Type):
                    self.write('this_ptr%%p%%%s = %s_ptr%%p' % (el.name, el.name))
                else:
                    self.write('%s_%s = %s_ptr%%p' % (t.name, el.name, el.name))
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
            if isinstance(t, Type):
                self.write('this_ptr = transfer(this, this_ptr)')
            if getset == "get":
                if isinstance(t, Type):
                    self.write('%s = this_ptr%%p%%%s' % (el.name, el.name))
                else:
                    self.write('%s = %s_%s' % (el.name, t.name, el.name))
            else:
                if isinstance(t, Type):
                    self.write('this_ptr%%p%%%s = %s' % (el.name, el.name))
                else:
                    self.write('%s_%s = %s' % (t.name, el.name, el.name))
        self.dedent()
        self.write('end subroutine %s%s__%s__%s' % (self.prefix, t.name, getset,
                                                    el.name))
        self.write()


