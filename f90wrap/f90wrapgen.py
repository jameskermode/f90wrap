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

import copy
import logging

import numpy as np

from f90wrap import fortran as ft
from f90wrap import codegen as cg

# numeric codes for Fortran types.
# Those with suffix _A are 1D arrays, _A2 are 2D arrays
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
        
    """
    def __init__(self, prefix, sizeof_fortran_t, string_lengths, abort_func,
                 kind_modules, kind_map):
        cg.CodeGenerator.__init__(self, indent=' ' * 4,
                               max_length=80,
                               continuation='&')
        ft.FortranVisitor.__init__(self)
        self.prefix = prefix
        self.sizeof_fortran_t = sizeof_fortran_t
        self.string_lengths = string_lengths
        self.abort_func = abort_func
        self.kind_modules = kind_modules
        self.kind_map = kind_map

    def visit_Root(self, node):
        """
        Write a wrapper for top-level procedures. 
        """
        self.code = []
        self.generic_visit(node)
        if len(self.code) > 0:
            f90_wrapper_file = open('%s%s.f90' % (self.prefix, 'toplevel'), 'w')
            f90_wrapper_file.write(str(self))
            f90_wrapper_file.close()

    def visit_Module(self, node):
        """
        Wrap modules. Each Fortran module generates one wrapper source file.
        
        Subroutines and elements within each module are properly wrapped.
        """
        logging.info('F90WrapperGenerator visiting module %s' % node.name)
        self.code = []
        self.generic_visit(node)

        for el in node.elements:
            dims = filter(lambda x: x.startswith('dimension'), el.attributes)
            if len(dims) == 0:  # proper scalar type (normal or derived)
                self._write_scalar_wrappers(node, el, self.sizeof_fortran_t)  # where to get sizeof_fortran_t from??
            elif el.type.startswith('type'):  # array of derived types
                self._write_dt_array_wrapper(node, el, dims, self.sizeof_fortran_t)  # where to get dims from?
            else:
                if 'parameter' not in el.attributes:
                    self._write_sc_array_wrapper(node, el, dims, self.sizeof_fortran_t)

        if len(self.code) > 0:
            f90_wrapper_file = open('%s%s.f90' % (self.prefix, node.name), 'w')
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
        class DuplicateSymbolError:
            pass
        
        all_uses = {}
        for (mod, only) in self.kind_modules.iteritems():
            try:
                if extra_uses_dict is not None:
                    for (new_mod, new_only) in extra_uses_dict.iteritems():
                        for symbol in only:
                            if (symbol in new_only or
                                ('%s_%s => %s' % (mod, symbol, symbol) in new_only)):
                                raise DuplicateSymbolError
            except DuplicateSymbolError:
                continue
            all_uses[mod] = only[:]

        node_uses = []
        if hasattr(node, 'uses'):
            for use in node.uses:
                if isinstance(use, basestring):
                    node_uses.append((use, None))
                else:
                    node_uses.append(use)
                                
        if extra_uses_dict is not None:
            for (mod, only) in extra_uses_dict.iteritems():
                node_uses.append((mod, only))

        if node_uses:
            for (mod, only) in node_uses:
                if mod in all_uses:
                    if only is None:
                        continue
                    for symbol in only:
                        if symbol not in all_uses[mod]:
                            all_uses[mod] += [symbol]
                elif only is not None:
                    all_uses[mod] = list(only)
                else:
                    all_uses[mod] = None
        
        for mod, only in all_uses.iteritems():
            if only is not None:
                self.write('use %s, only: %s' % (mod, ', '.join(only)))
            else:
                self.write('use %s' % mod)

    def write_type_lines(self, tname):
        """
        Write a pointer type for a given type name
        
        Parameters
        ----------
        tname : `str`
            Should be the name of a derived type in the wrapped code.
        """
        tname = ft.strip_type(tname)
        self.write("""type %(typename)s_ptr_type
    type(%(typename)s), pointer :: p => NULL()
end type %(typename)s_ptr_type""" % {'typename': tname})

    def write_arg_decl_lines(self, node):
        """
        Write argument declaration lines to the code
        
        Takes care of argument attributes, and opaque references for derived 
        types, as well as f2py-specific lines.
        """
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
        """
        Write transfer of opaque references.
        """
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
        """
        Write special user-provided init lines to a node.
        """
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
        """
        Properly write function/subroutine calls
        """
        if 'skip_call' in node.attributes:
            return

        if hasattr(node, 'orig_node'):
            node = node.orig_node

        def dummy_arg_name(arg):
            return arg.name

        def actual_arg_name(arg):
            if ((hasattr(node, 'transfer_in') and arg.name in node.transfer_in) or
                (hasattr(node, 'transfer_out') and arg.name in node.transfer_out)):
                return '%s_ptr%%p' % arg.name
            else:
                return arg.name

        if node.mod_name is not None:
            # use keyword arguments if subroutine is in a module and we have an explicit interface
            arg_names = ['%s=%s' % (dummy_arg_name(arg), actual_arg_name(arg)) for arg in node.arguments
                        if 'intent(hide)' not in arg.attributes]
        else:
            arg_names = [actual_arg_name(arg) for arg in node.arguments if 'intent(hide)' not in arg.attributes]

        if isinstance(node, ft.Function):
            self.write('%(ret_val)s = %(func_name)s(%(arg_names)s)' %
                       {'ret_val': node.ret_val.name,
                        'func_name': node.name,
                        'arg_names': ', '.join(arg_names)})
        else:
            self.write('call %(sub_name)s(%(arg_names)s)' %
                       {'sub_name': node.name,
                        'arg_names': ', '.join(arg_names)})

    def write_transfer_out_lines(self, node):
        """
        WRite transfer from opaque reference.
        """
        for arg in node.arguments:
            if arg.name in node.transfer_out:
                self.write('%(arg_name)s = transfer(%(arg_name)s_ptr, %(arg_name)s)' %
                           {'arg_name': arg.name})

    def write_finalise_lines(self, node):
        """
        Deallocate the opaque reference to clean up.
        """
        for dealloc in node.deallocate:
            self.write('deallocate(%s_ptr%%p)' % dealloc)  # (self.prefix, dealloc))

    def visit_Subroutine(self, node):
        """
        Visit a Subroutine node of the parse tree.
        
        Writes the code necessary for each propery wrapped subroutine.
        """
        logging.info('F90WrapperGenerator visiting subroutine %s' % node.name)
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
        """
        Properly wraps derived types, including derived-type arrays.
        """
        logging.info('F90WrapperGenerator visiting type %s' % node.name)

        for el in node.elements:
            dims = filter(lambda x: x.startswith('dimension'), el.attributes)
            if len(dims) == 0:  # proper scalar type (normal or derived)
                self._write_scalar_wrappers(node, el, self.sizeof_fortran_t)  # where to get sizeof_fortran_t from??
            elif el.type.startswith('type'):  # array of derived types
                self._write_dt_array_wrapper(node, el, dims, self.sizeof_fortran_t)  # where to get dims from?
            else:
                self._write_sc_array_wrapper(node, el, dims, self.sizeof_fortran_t)

        return self.generic_visit(node)

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
        # The following maps the element type to a numeric code
        fortran_type_code = {'d': T_REAL_A,
                             'i': T_INTEGER_A,
                             'S': T_CHAR_A,
                             'complex': T_COMPLEX_A}

        numpy_type_map = {'real(8)': 'd',  # FIXME user-provided kind_modules should be included here
                          'real(dp)':'d',
                          'real(dl)':'d',
                          'integer':'i',
                          'logical':'i',
                          'double precision': 'd',
                          'character*(*)':'S',
                          'complex(dp)':'complex',
                          'real(16)':'float128',
                          'real(qp)':'float128',
                          'real(kind=dp)': 'd'}

        typename = numpy_type_map.get(el.type, el.type)

        if isinstance(t, ft.Type):
            this = 'this, '
        else:
            this = 'dummy_this, '

        self.write('subroutine %s%s__array__%s(%snd, dtype, dshape, dloc)' % (self.prefix, t.name, el.name, this))
        self.indent()

        if isinstance(t, ft.Module):
            self.write_uses_lines(t, {t.name: ['%s_%s => %s' % (t.name, el.name, el.name)]})
        else:
            self.write_uses_lines(t)
            
        self.write('implicit none')
        if isinstance(t, ft.Type):
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
        if isinstance(t, ft.Type):
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

    def _write_dt_array_wrapper(self, t, element, dims,
                               sizeof_fortran_t):
        """
        Write fortran get/set/len routines for a derived-type array.
        
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
        if element.type.startswith('type') and len(dims) != 1:
            return

        self._write_array_getset_item(t, element, sizeof_fortran_t, 'get')
        self._write_array_getset_item(t, element, sizeof_fortran_t, 'set')
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
        if 'parameter' not in element.attributes:
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
            self.write('subroutine %s%s__array_%sitem__%s(this, i, %s)' % (self.prefix, t.name,
                                                                           getset, el.name,
                                                                           el.name))
        else:
            self.write('subroutine %s%s__array_%sitem__%s(i, %s)' % (self.prefix, t.name,
                                                                     getset, el.name,
                                                                     el.name))
            
        self.indent()
        self.write()
        if isinstance(t, ft.Module):
            self.write_uses_lines(t, {t.name: ['%s_%s => %s' % (t.name, el.name, el.name)]})
        else:
            self.write_uses_lines(t)
        self.write('implicit none')
        self.write()
        if isinstance(t, ft.Type):
            self.write_type_lines(t.name)
        self.write_type_lines(el.type)

        if isinstance(t, ft.Type):
            self.write('integer, intent(in) :: this(%d)' % sizeof_fortran_t)
            self.write('type(%s_ptr_type) :: this_ptr' % t.name)
            array_name = 'this_ptr%%p%%%s' % el.name
        else:
            array_name = '%s_%s' % (t.name, el.name)            
        self.write('integer, intent(in) :: i')
        self.write('integer, intent(%s) :: %s(%d)' % (inout, el.name, sizeof_fortran_t))
        self.write('type(%s_ptr_type) :: %s_ptr' % (ft.strip_type(el.type), el.name))
        self.write()
        if isinstance(t, ft.Type):
            self.write('this_ptr = transfer(this, this_ptr)')

        if 'allocatable' in el.attributes:
            self.write('if (allocated(%s)) then' % array_name)
            self.indent()

        self.write('if (i < 1 .or. i > size(%s)) then' % array_name)
        self.indent()
        self.write('call %s("array index out of range")' % self.abort_func)
        self.dedent()
        self.write('else')
        self.indent()

        if getset == "get":
            self.write('%s_ptr%%p => %s(i)' % (el.name, array_name))
            self.write('%s = transfer(%s_ptr,%s)' % (el.name, el.name, el.name))
        else:
            self.write('%s_ptr = transfer(%s,%s_ptr)' % (el.name, el.name, el.name))
            self.write('%s(i) = %s_ptr%%p' % (array_name, el.name))

        self.dedent()
        self.write('endif')

        if 'allocatable' in el.attributes:
            self.dedent()
            self.write('else')
            self.indent()
            self.write('call %s("derived type array not allocated")' % self.abort_func)
            self.dedent()
            self.write('end if')

        self.dedent()
        self.write('end subroutine %s%s__array_%sitem__%s' % (self.prefix, t.name,
                                                              getset, el.name))
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
            self.write('subroutine %s%s__array_len__%s(this, n)' % (self.prefix, t.name, el.name))
        else:
            self.write('subroutine %s%s__array_len__%s(n)' % (self.prefix, t.name, el.name))
        self.indent()
        self.write()
        if isinstance(t, ft.Module):
            self.write_uses_lines(t, {t.name: ['%s_%s => %s' % (t.name, el.name, el.name)]})
        else:
            self.write_uses_lines(t)
        self.write('implicit none')
        self.write()
        if isinstance(t, ft.Type):
            self.write_type_lines(t.name)
        self.write_type_lines(el.type)
        self.write('integer, intent(out) :: n')
        if isinstance(t, ft.Type):
            self.write('integer, intent(in) :: this(%d)' % sizeof_fortran_t)
            self.write('type(%s_ptr_type) :: this_ptr' % t.name)
            self.write()
            self.write('this_ptr = transfer(this, this_ptr)')
            array_name = 'this_ptr%%p%%%s' % el.name
        else:
            array_name = '%s_%s' % (t.name, el.name)

        if 'allocatable' in el.attributes:
            self.write('if (allocated(%s)) then' % array_name)
            self.indent()

        self.write('n = size(%s)' % array_name)

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
        # getset and inout just change things simply from a get to a set routine.
        inout = "in"
        if getset == "get":
            inout = "out"

        if isinstance(t, ft.Type):
            this = 'this, '
        elif isinstance(t, ft.Module):
            this = ''
        else:
            raise ValueError("Don't know how to write scalar wrappers for %s type %s" (t, type(t)))

        self.write('subroutine %s%s__%s__%s(%s%s)' % (self.prefix, t.name,
                                                    getset, el.name, this, el.name))
        self.indent()
        if isinstance(t, ft.Module):
            self.write_uses_lines(t, {t.name: ['%s_%s => %s' % (t.name, el.name, el.name)]})
        else:
            self.write_uses_lines(t)

        self.write('implicit none')
        if isinstance(t, ft.Type):
            self.write_type_lines(t.name)

        if el.type.startswith('type'):
            self.write_type_lines(el.type)

        if isinstance(t, ft.Type):
            self.write('integer, intent(in)   :: this(%d)' % sizeof_fortran_t)
            self.write('type(%s_ptr_type) :: this_ptr' % t.name)

        # Return/set by value
        attributes = el.attributes[:]
        for key in ['pointer', 'allocatable', 'public', 'parameter', 'save']:
            if key in attributes:
                attributes.remove(key)

        if el.type.startswith('type'):
            # For derived types elements, treat as opaque reference
            self.write('integer, intent(%s) :: %s(%d)' % (inout, el.name, sizeof_fortran_t))

            self.write('type(%s_ptr_type) :: %s_ptr' % (ft.strip_type(el.type), el.name))
            self.write()
            if isinstance(t, ft.Type):
                self.write('this_ptr = transfer(this, this_ptr)')
            if getset == "get":
                if isinstance(t, ft.Type):
                    self.write('%s_ptr%%p => this_ptr%%p%%%s' % (el.name, el.name))
                else:
                    self.write('%s_ptr%%p => %s_%s' % (el.name, t.name, el.name))
                self.write('%s = transfer(%s_ptr,%s)' % (el.name, el.name, el.name))
            else:
                self.write('%s_ptr = transfer(%s,%s_ptr)' % (el.name,
                                                             el.name,
                                                             el.name))
                if isinstance(t, ft.Type):
                    self.write('this_ptr%%p%%%s = %s_ptr%%p' % (el.name, el.name))
                else:
                    self.write('%s_%s = %s_ptr%%p' % (t.name, el.name, el.name))
        else:
            if attributes != []:
                self.write('%s, %s, intent(%s) :: %s' % (el.type,
                                                         ','.join(attributes),
                                                         inout, el.name))
            else:
                self.write('%s, intent(%s) :: %s' % (el.type, inout, el.name))
            self.write()
            if isinstance(t, ft.Type):
                self.write('this_ptr = transfer(this, this_ptr)')
            if getset == "get":
                if isinstance(t, ft.Type):
                    self.write('%s = this_ptr%%p%%%s' % (el.name, el.name))
                else:
                    self.write('%s = %s_%s' % (el.name, t.name, el.name))
            else:
                if isinstance(t, ft.Type):
                    self.write('this_ptr%%p%%%s = %s' % (el.name, el.name))
                else:
                    self.write('%s_%s = %s' % (t.name, el.name, el.name))
        self.dedent()
        self.write('end subroutine %s%s__%s__%s' % (self.prefix, t.name, getset,
                                                    el.name))
        self.write()


