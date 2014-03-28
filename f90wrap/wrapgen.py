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

class F90WrapperGenerator(FortranVisitor, CodeGenerator):

    def __init__(self, prefix):
        CodeGenerator.__init__(self, indent='    ',
                               max_length=80,
                               continuation='&')
        FortranVisitor.__init__(self)
        self.prefix = prefix

    def visit_Module(self, node):
        self.code = []
        self.generic_visit(node)
        f90_wrapper_file = open('%s%s.f90' % (self.prefix, node.name), 'w')
        f90_wrapper_file.write(str(self))
        f90_wrapper_file.close()

    def write_uses_lines(self, node):
        self.write('! BEGIN write_uses_lines')
        for (mod, only) in node.uses:
            if only is not None:
                self.write('use %s, only: %s' % (mod, ' '.join(only)))
            else:
                self.write('use %s' % mod)
        self.write('! END write_uses_lines')
        self.write()

    def write_type_lines(self, node):
        self.write('! BEGIN write_type_lines')
        for typename in node.types:
            self.write("""type %(typename)s_ptr_type
    type(%(typename)s), pointer :: p => NULL()
end type %(typename)s_ptr_type""" % {'typename': typename})
        self.write('! END write_type_lines')
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

        self.write("subroutine %(sub_name)s(%(arg_names)s)" %
                   {'sub_name': self.prefix + node.name,
                    'arg_names': ', '.join([arg.name for arg in node.arguments])})
        self.indent()
        self.write()
        self.write_uses_lines(node)
        self.write("implicit none")
        self.write()
        self.write_type_lines(node)
        self.write_arg_decl_lines(node)
        self.write_transfer_in_lines(node)
        self.write_init_lines(node)
        self.write_call_lines(node)
        self.write_transfer_out_lines(node)
        self.write_finalise_lines(node)
        self.dedent()
        self.write("end subroutine %(sub_name)s" % {'sub_name': self.prefix + node.name})
        self.write()
        self.write()


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

            typename = arg.type[5:-1]
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
        """Replace interfaces with one procedure by that procedure"""
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



def transform_to_f90_wrapper(tree, types, kinds, callbacks, constructors,
                             destructors, short_names, init_lines,
                             string_lengths, default_string_length,
                             sizeof_fortran_t):
    """
    Apply a number of rules to *tree* to make it suitable for passing to
    a F90WrapperGenerator's visit() method. Transformations performed are:

     * Removal of private symbols
     * Removal of unwrappable routines and optional arguments
     * Conversion of all functions to subroutines
     * Update of subroutine uses clauses
     * Conversion of derived type arguments to opaque integer arrays
       via Fortran transfer() intrinsic.
     * ...
    """

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
