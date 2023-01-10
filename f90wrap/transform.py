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

from __future__ import print_function

import copy
import logging
import re
import hashlib

from f90wrap import fortran as ft

log = logging.getLogger(__name__)


class AccessUpdater(ft.FortranTransformer):
    """Visit module contents and update public_symbols and
       private_symbols lists to be consistent with (i) default module
       access; (ii) public and private statements at module level;
       (iii) public and private statement in types; (iv) public
       and private attributes of individual elements."""

    def __init__(self, force_public=None):
        self.mod = None
        self.type = None
        self.force_public = ()
        if force_public is not None:
            self.force_public = force_public

    def update_access(self, node, mod, default_access, in_type=False):
        if node.name in self.force_public:
            log.debug('marking public symbol (forced) ' + node.name)
            mod.public_symbols.append(node.name)
            if 'private' in getattr(node, 'attributes', []):
                node.attributes.remove('private')
                node.attributes.append('public')

        if default_access == 'public':
            if ('private' not in getattr(node, 'attributes', []) and
                        node.name not in mod.private_symbols):

                # symbol should be marked as public if it's not already
                if not in_type and node.name not in mod.public_symbols:
                    log.debug('marking public symbol ' + node.name)
                    mod.public_symbols.append(node.name)
            else:
                # symbol should be marked as private if it's not already
                if not in_type and (node.name not in mod.private_symbols and
                                            'callback' not in getattr(node, 'attributes', [])):
                    log.debug('marking private symbol ' + node.name)
                    mod.private_symbols.append(node.name)

        elif default_access == 'private':
            if ('public' not in getattr(node, 'attributes', []) and
                        node.name not in mod.public_symbols):

                # symbol should be marked as private if it's not already
                if not in_type and (node.name not in mod.private_symbols and
                                            'callback' not in getattr(node, 'attributes', [])):
                    log.debug('marking private symbol ' + node.name)
                    mod.private_symbols.append(node.name)
            else:
                # symbol should be marked as public if it's not already
                if not in_type and node.name not in mod.public_symbols:
                    log.debug('marking public symbol ' + node.name)
                    mod.public_symbols.append(node.name)

        else:
            raise ValueError('bad default access %s for reference %s' %
                             (mod.default_access, mod.name))

    def visit_Module(self, mod):
        # keep track of the current module
        self.mod = mod
        mod = self.generic_visit(mod)
        self.mod = None
        return mod

    def visit_Procedure(self, node):
        if self.mod is None:
            return self.generic_visit(node)
        self.update_access(node, self.mod, self.mod.default_access)
        return self.generic_visit(node)

    def visit_Interface(self, node):
        if self.mod is None:
            return self.generic_visit(node)
        self.update_access(node, self.mod, self.mod.default_access)
        return self.generic_visit(node)

    def visit_Type(self, node):
        if self.mod is None:
            return self.generic_visit(node)
        self.type = node
        self.update_access(node, self.mod, self.mod.default_access)
        node.default_access = 'public'
        if 'private' in node.attributes:
            node.default_access = 'private'
        node = self.generic_visit(node)
        self.type = None
        return node

    def visit_Element(self, node):
        if self.type is not None:
            self.update_access(node, self.mod, self.type.default_access, in_type=True)
        else:
            self.update_access(node, self.mod, self.mod.default_access)
        return node


class PrivateSymbolsRemover(ft.FortranTransformer):
    """
    Transform a tree by removing private symbols
    """

    def __init__(self):
        self.mod = None

    def visit_Module(self, mod):
        # keep track of the current module
        self.mod = mod
        mod = self.generic_visit(mod)
        self.mod = None
        return mod

    def visit_Procedure(self, node):
        if self.mod is None:
            return self.generic_visit(node)

        if node.name in self.mod.private_symbols:
            log.debug('removing private symbol %s' % node.name)
            return None

        if hasattr(node, 'attributes') and 'private' in node.attributes:
            log.debug('removing private symbol by attribute list %s' % node.name)
            return None

        return self.generic_visit(node)

    def visit_Interface(self, node):
        # remove entirely private interfaces
        if node.name in self.mod.private_symbols:
            log.debug('removing private symbol on interface %s' % node.name)
            return None

        # do not call generic_visit(), so we don't
        # remove private procedures within public
        # interfaces, as these should still be wrapped
        return node

    visit_Type = visit_Procedure
    visit_Element = visit_Procedure


def remove_private_symbols(node, force_public=None):
    """
    Walk the tree starting at *node*, removing all private symbols.

    This function first applies the AccessUpdater transformer to
    ensure module *public_symbols* and *private_symbols* are up to
    date with *default_access* and individual `public` and `private`
    attributes.
    """

    node = AccessUpdater(force_public).visit(node)
    node = PrivateSymbolsRemover().visit(node)
    return node


class UnwrappablesRemover(ft.FortranTransformer):
    def __init__(self, callbacks, types, constructors, destructors, remove_optional_arguments):
        self.callbacks = callbacks
        self.types = types
        self.constructors = constructors
        self.destructors = destructors
        self.remove_optional_arguments = remove_optional_arguments

    def visit_Interface(self, node):
        # don't wrap operator overloading routines
        if node.name.startswith('operator('):
            return None

        return self.generic_visit(node)

    def visit_Binding(self, node):
        # Remove unwrapable procedures from bindings
        new_procs = []
        for proc in node.procedures:
            new_proc = self.visit_Procedure(proc)
            if new_proc:
                new_procs.append(new_proc)
        node.procedures = new_procs
        if new_procs:
            return node
        else:
            return None  # Binding is empty; remove from type

    def visit_Procedure(self, node):
        # special case: keep all constructors and destructors, although
        # they may have pointer arguments
        for suff in self.constructors + self.destructors:
            if node.name.endswith(suff):
                return self.generic_visit(node)

        # don't wrap operator overloading routines
        if node.name.startswith('operator('):
            return None

        # FIXME don't wrap callback arguments
        if 'callback' in node.attributes:
            return None

        args = node.arguments[:]
        if isinstance(node, ft.Function):
            args.append(node.ret_val)
        for arg in args:
            # only callback functions in self.callbacks
            if 'callback' in arg.attributes:
                if node.name not in self.callbacks:
                    log.warning('removing callback routine %s' % node.name)
                    return None
                else:
                    continue

            if 'optional' in arg.attributes:
                # we can remove the argument instead of the whole routine
                # fortran permits opt arguments before compulsory ones, so continue not return
                # generic_visit is done later on anyways
                continue
            else:
                # no allocatables or pointers
                if 'allocatable' in arg.attributes or 'pointer' in arg.attributes:
                    log.warning('removing routine %s due to allocatable/pointer arguments' % node.name)
                    return None

                dims = [attrib for attrib in arg.attributes if attrib.startswith('dimension')]

                # # no complex scalars (arrays are OK)
                # if arg.type.startswith('complex') and len(dims) == 0:
                #    log.debug('removing routine %s due to complex scalar arguments' % node.name)
                #    return None

                # no derived types apart from those in self.types
                typename = ft.derived_typename(arg.type)
                if typename and typename not in self.types:
                    log.warning('removing routine %s due to unsupported derived type %s' %
                                  (node.name, arg.type))
                    return None

                # no arrays of derived types of assumed shape, or more than one dimension
                # Yann - EXPERIMENTAL !!
                if typename and len(dims) != 0:
                    # log.warning('removing routine %s due to unsupported arrays of derived types %s' %
                    #               (node.name, arg.type))
                    # return none
                    if len(dims) > 1:
                        raise ValueError('more than one dimension attribute found for arg %s' % arg.name)
                    dimensions_list = ArrayDimensionConverter.split_dimensions(dims[0])
                    if len(dimensions_list) > 1 or ':' in dimensions_list:
                        log.warning('removing routine %s due to derived type array argument : %s -- currently, only '
                                     'fixed-lengh one-dimensional arrays of derived type are supported'
                                     % (node.name, arg.name))
                        return None

        return self.generic_visit(node)

    def visit_Argument(self, node):
        if not hasattr(node, 'attributes'):
            return self.generic_visit(node)

        if not 'optional' in node.attributes:
            return self.generic_visit(node)

        if node.name in self.remove_optional_arguments:
            log.warning('removing optional argument %s' %
                         node.name)
            return None

        # remove optional allocatable/pointer arguments
        if 'allocatable' in node.attributes or 'pointer' in node.attributes:
            log.warning('removing optional argument %s due to allocatable/pointer attributes' %
                         node.name)
            return None

        dims = [attrib for attrib in node.attributes if attrib.startswith('dimension')]

        # remove optional complex scalar arguments
        if node.type.startswith('complex') and len(dims) == 0:
            log.warning('removing optional argument %s as it is a complex scalar' % node.name)
            return None

        # remove optional derived types not in self.types
        typename = ft.derived_typename(node.type)
        if typename and typename not in self.types:
            log.warning('removing optional argument %s due to unsupported derived type %s' %
                          (node.name, node.type))
            return None

        # remove optional arrays of derived types
        # EXPERIMENTAL !
        if typename and len(dims) != 0:
            if len(dims) > 1:
                raise ValueError('more than one dimension attribute found for arg %s' % node.name)
            dimensions_list = ArrayDimensionConverter.split_dimensions(dims[0])
            if len(dimensions_list) > 1 or ':' in dimensions_list:
                log.warning(
                    'test removing optional argument %s as only one dimensional fixed-length arrays are currently supported for derived type %s array' %
                    (node.name, node.type))
                return None

        return self.generic_visit(node)

    def visit_Type(self, node):
        """
        Remove unwrappable elements inside derived types
        """
        if node.name not in self.types:
            log.warning('removing type %s' % node.name)
            return None
        else:
            elements = []
            for element in node.elements:
                # Get the number of dimensions of the element (if any)
                dims = [attr for attr in element.attributes if attr.startswith(
                    'dimension')]  # dims = filter(lambda x: x.startswith('dimension'), element.attributes) provides a filter object, so dims == [] would ALWAYS be false
                if element.type.lower() == 'type(c_ptr)':
                    log.warning('removing %s.%s as type(c_ptr) unsupported' %
                                  (node.name, element.name))
                    continue
                typename = ft.derived_typename(element.type)
                if typename  and typename not in self.types:
                    log.warning('removing %s.%s as type %s unsupported' %
                                  (node.name, element.name, element.type))
                    continue
                elements.append(element)
            node.elements = elements
            return self.generic_visit(node)

    def visit_Module(self, node):
        """
        Remove unwrappable elements inside modules.

        As above, but also includes derived type elements from modules
        that do not have the "target" attribute
        """
        elements = []
        for element in node.elements:
            # Get the number of dimensions of the element (if any)
            dims = [attr for attr in element.attributes if attr.startswith(
                'dimension')]  # filter(lambda x: x.startswith('dimension'), element.attributes) provides a filter object, so dims == [] would ALWAYS be false
            if 'pointer' in element.attributes and dims != []:
                log.warning('removing %s.%s due to pointer attribute' %
                              (node.name, element.name))
                continue
            if element.type.lower() == 'type(c_ptr)':
                log.warning('removing %s.%s as type(c_ptr) unsupported' %
                              (node.name, element.name))
                continue
            typename = ft.derived_typename(element.type)
            if typename and 'target' not in element.attributes:
                log.warning('removing %s.%s as missing "target" attribute' %
                              (node.name, element.name))
                continue
            if typename and typename not in self.types:
                log.warning('removing %s.%s as type %s unsupported' %
                              (node.name, element.name, element.type))
                continue
            # parameter ARRAYS in modules live only in the mind of the compiler
            if 'parameter' in element.attributes and dims != []:
                log.warning('removing %s.%s as it has "parameter array" attribute' %
                              (node.name, element.name))
                continue

            elements.append(element)
        node.elements = elements
        return self.generic_visit(node)


def fix_subroutine_uses_clauses(tree, types):
    """Walk over all nodes in tree, updating subroutine uses
       clauses to include the parent module and all necessary
       modules from types

       Also rename any arguments that clash with module names.
    """

    for mod, sub, arguments in ft.walk_procedures(tree):
        sub.uses = set()
        if mod is not None:
            sub_name = sub.name
            if hasattr(sub, 'call_name'):
                sub_name = sub.call_name
            if sub.mod_name is None:
                sub.mod_name = mod.name
            sub.uses.add((sub.mod_name, (sub_name,)))

        for arg in arguments:
            typename = ft.strip_type(arg.type)
            if typename and typename in types:
                sub.uses.add((types[ft.strip_type(arg.type)].mod_name, (ft.strip_type(arg.type),)))

    for mod, sub, arguments in ft.walk_procedures(tree):
        for arg in arguments:
            for (mod_name, type_name) in sub.uses:
                if arg.name == mod_name:
                    arg.name += '_'

    return tree


def fix_element_uses_clauses(tree, types):
    """
    Add uses clauses to derived type elements in modules
    """
    for mod in ft.walk_modules(tree):
        for el in mod.elements:
            el.uses = set()
            typename = ft.derived_typename(el.type)
            if typename and typename in types:
                el.uses.add((types[el.type].mod_name, (typename,)))

    return tree


def set_intent(attributes, intent):
    """Remove any current "intent" from attributes and replace with intent given"""
    attributes = [attr for attr in attributes if not attr.startswith('intent')]
    attributes.append(intent)
    return attributes


def convert_derived_type_arguments(tree, init_lines, sizeof_fortran_t):
    for mod, sub, arguments in ft.walk_procedures(tree, include_ret_val=True):
        sub.types = set()
        sub.transfer_in = []
        sub.transfer_out = []
        sub.allocate = []
        sub.deallocate = []

        if 'constructor' in sub.attributes:
            sub.arguments[0].attributes = set_intent(sub.arguments[0].attributes, 'intent(out)')

        if 'destructor' in sub.attributes:
            log.debug('deallocating arg "%s" in %s', sub.arguments[0].name, sub.name)
            sub.deallocate.append(sub.arguments[0].name)

        for arg in arguments:
            if not hasattr(arg, 'type') or not ft.is_derived_type(arg.type):
                continue

            # save original Fortran intent since we'll be overwriting it
            # with intent of the opaque pointer
            arg.attributes = arg.attributes + ['fortran_' + attr for attr in
                                               arg.attributes if attr.startswith('intent')]

            typename = ft.strip_type(arg.type)
            arg.wrapper_type = 'integer'
            arg.wrapper_dim = sizeof_fortran_t
            sub.types.add(typename)

            if typename in init_lines:
                use, (exe, exe_optional) = init_lines[typename]
                if use is not None:
                    sub.uses.add(use)
                arg.init_lines = (exe_optional, exe)

            if 'intent(out)' in arg.attributes:
                arg.attributes = set_intent(arg.attributes, 'intent(out)')
                sub.transfer_out.append(arg.name)
                if 'pointer' not in arg.attributes:
                    log.debug('allocating arg "%s" in %s' % (arg.name, sub.name))
                    sub.allocate.append(arg.name)
            else:
                arg.attributes = set_intent(arg.attributes, 'intent(in)')
                sub.transfer_in.append(arg.name)

    return tree


def convert_array_intent_out_to_intent_inout(tree):
    """
    Find all intent(out) array arguments and convert to intent(inout)
    """
    for mod, sub, arguments in ft.walk_procedures(tree, include_ret_val=True):
        if '__array__' in sub.name:
            # special case for array wrappers, which shouldn't be touched
            continue
        for arg in arguments:
            dims = [attr for attr in arg.attributes if attr.startswith('dimension')]
            if dims != [] or 'optional' in arg.attributes:
                if dims != [] and len(dims) != 1:
                    raise ValueError('more than one dimension attribute found for arg %s' % arg.name)
                if 'intent(out)' in arg.attributes:
                    arg.attributes = set_intent(arg.attributes, 'intent(inout)')
    return tree


class StringLengthConverter(ft.FortranVisitor):
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
                string_length = self.default_string_length

        except ValueError:
            string_length = 1

        node.type = 'character*(%s)' % str(string_length)


class ArrayDimensionConverter(ft.FortranVisitor):
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

    valid_dim_re = re.compile(r'^(([-0-9.e]+)|(size\([_a-zA-Z0-9\+\-\*\/,]*\))|(len\(.*\)))$')

    @staticmethod
    def split_dimensions(dim):
        """Given a string like "dimension(a,b,c)" return the list of dimensions ['a','b','c']."""
        dim = dim[10:-1]  # remove "dimension(" and ")"
        br = 0
        d = 1
        ds = ['']
        for c in dim:
            if c != ',': ds[-1] += c
            if c == '(':
                br += 1
            elif c == ')':
                br -= 1
            elif c == ',':
                if br == 0:
                    ds.append('')
                else:
                    ds[-1] += ','
        return ds

    def visit_Procedure(self, node):

        n_dummy = 0
        for arg in node.arguments:
            dims = [attr for attr in arg.attributes if attr.startswith('dimension')]
            if dims == []:
                continue
            if len(dims) != 1:
                raise ValueError('more than one dimension attribute found for arg %s' % arg.name)

            ds = ArrayDimensionConverter.split_dimensions(dims[0])

            new_dummy_args = []
            new_ds = []
            for i, d in enumerate(ds):
                if ArrayDimensionConverter.valid_dim_re.match(d):
                    if d.startswith('len'):
                        arg.f2py_line = ('!f2py %s %s, dimension(%s) :: %s' % \
                                         (arg.type,
                                          ','.join(
                                              [attr for attr in arg.attributes if not attr.startswith('dimension')]),
                                          d.replace('len', 'slen'), arg.name))
                    new_ds.append(d)
                    continue
                dummy_arg = ft.Argument(name='n%d' % n_dummy, type='integer', attributes=['intent(hide)'])

                if 'intent(out)' not in arg.attributes:
                    dummy_arg.f2py_line = ('!f2py intent(hide), depend(%s) :: %s = shape(%s,%d)' %
                                           (arg.name, dummy_arg.name, arg.name, i))
                new_dummy_args.append(dummy_arg)
                new_ds.append(dummy_arg.name)
                n_dummy += 1

            if new_dummy_args != []:
                log.debug('adding dummy arguments %r to %s' % (new_dummy_args, node.name))
                arg.attributes = ([attr for attr in arg.attributes if not attr.startswith('dimension')] +
                                  ['dimension(%s)' % ','.join(new_ds)])
                node.arguments.extend(new_dummy_args)


class MethodFinder(ft.FortranTransformer):
    def __init__(self, types, constructor_names, destructor_names, short_names,
                 move_methods, shorten_routine_names=True, modules_for_type=None):
        self.types = types
        self.constructor_names = constructor_names
        self.destructor_names = destructor_names
        self.short_names = short_names
        self.move_methods = move_methods
        self.shorten_routine_names = shorten_routine_names
        self.modules_for_type = modules_for_type

    def visit_Interface(self, node):
        new_procs = []
        for proc in node.procedures:
            if isinstance(proc, ft.Procedure):
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
                (node.arguments[0] is not None and
                         node.arguments[0].type not in self.types)):
            # procedure is not a method, so leave it alone
            return node

        typ = self.types[node.arguments[0].type]
        node.method_name = node.name
        if self.shorten_routine_names:
            # remove prefix from subroutine name to get method name
            prefices = [typ.name + '_']
            if typ.name in self.short_names:
                prefices.append(self.short_names[typ.name] + '_')
            for prefix in prefices:
                if node.name.startswith(prefix):
                    node.method_name = node.name[len(prefix):]

        # label constructors and destructors
        if any(node.method_name.endswith(dn) for dn in self.destructor_names):
            node.attributes.append('destructor')
        elif any(node.method_name.endswith(cn) for cn in self.constructor_names):
            node.attributes.append('constructor')

        if (self.move_methods or
                    'constructor' in node.attributes or
                    'destructor' in node.attributes):

            node.attributes.append('method')
            node.type_name = typ.name

            if interface is None:
                if node.mod_name not in self.modules_for_type[typ.mod_name]:
                    # procedure looks like a method but
                    # defined in a different module - leave it where it is
                    log.info(f'Not moving method {node.name} from module {node.mod_name} as '
                             f'type defined in different module {self.modules_for_type[typ.mod_name]}')
                    return node

                # just a regular method - move into typ.procedures
                typ.procedures.append(node)
                log.info('added method %s to type %s' %
                              (node.method_name, typ.name))
            else:
                # this method was originally inside an interface,
                # so we need to replicate Interface inside the Type
                for intf in typ.interfaces:
                    if intf.name == interface.name:
                        intf.procedures.append(node)
                        log.info('added method %s to interface %s in type %s module %r' %
                                      (node.method_name, intf.name, typ.name, node.mod_name))
                        break
                else:
                    intf = ft.Interface(interface.name,
                                        interface.filename,
                                        interface.doc,
                                        interface.lineno,
                                        [node],
                                        mod_name=node.mod_name,
                                        type_name=typ.name)
                    typ.interfaces.append(intf)
                    log.info('added method %s to new interface %s in type %s module %r' %
                                  (node.method_name, intf.name, typ.name, node.mod_name))

            # remove method from parent since we've added it to Type
            return None
        else:
            return node


class ConstructorExcessToClassMethod(ft.FortranTransformer):
    """ Handle classes with multiple constructors

    Count the number of constructors per class and choose only one.
    Method of choice: the one with the shortest name.
    The rest are relabeled to classmethods"""

    def visit_Type(self, node):
        log.debug('visiting %r' % node)

        constructor_names = []

        for child in node.procedures:
            if 'constructor' in child.attributes:
                constructor_names.append(child.name)

        log.info('visiting %r found %d constructors with names: %s', node, len(constructor_names), constructor_names)

        if len(constructor_names) > 1:
            # now we need to modify all but one to classmethods
            # for now, we are taking the one with the shortest names
            # fixme: make this more general and possible to set up
            chosen = min(constructor_names, key=len)

            for child in node.procedures:
                if 'constructor' in child.attributes:
                    if child.name == chosen:
                        log.info('found multiple constructors, chose shortest named %s', child.name)
                    else:
                        #child.attributes = list(set(child.attributes))  # fixme: decide if this is needed at all
                        #child.attributes.remove('constructor')
                        child.attributes.append('classmethod')
                        log.info('transform excess constructor to classmethod %s', child.name)

        return self.generic_visit(node)

    visit_Module = visit_Type


def collapse_single_interfaces(tree):
    """Collapse interfaces which contain only a single procedure."""

    class _InterfaceCollapser(ft.FortranTransformer):
        """Replace interfaces with only one procedure by that procedure"""

        def visit_Interface(self, node):
            if len(node.procedures) == 1:
                proc = node.procedures[0]
                proc.doc = node.doc + proc.doc
                log.debug('collapsing single-component interface %s' % proc.name)
                return proc
            else:
                return node

    class _ProcedureRelocator(ft.FortranTransformer):
        """Filter interfaces and procedures into correct lists"""

        def visit_Type(self, node):
            log.debug('visiting %r' % node)
            interfaces = []
            procedures = []
            for child in ft.iter_child_nodes(node):
                if isinstance(child, ft.Interface):
                    interfaces.append(child)
                elif isinstance(child, ft.Procedure):
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
    for node in ft.walk(tree):
        if not isinstance(node, ft.Type):
            continue
        for child in ft.iter_child_nodes(node):
            if 'constructor' in child.attributes:
                log.info('found constructor %s', child.name)
                break
        else:

            log.info('adding missing constructor for %s', node.name)
            new_node = ft.Subroutine('%s_initialise' % node.name,
                                     node.filename,
                                     ['Automatically generated constructor for %s' % node.name],
                                     node.lineno,
                                     [ft.Argument(name='this',  # self.prefix + 'this' would probably be safer
                                                  filename=node.filename,
                                                  doc=['Object to be constructed'],
                                                  lineno=node.lineno,
                                                  attributes=['intent(out)'],
                                                  type='type(%s)' % node.name)],
                                     node.uses,
                                     ['constructor', 'skip_call'],
                                     mod_name=node.mod_name,
                                     type_name=node.name)
            new_node.method_name = '__init__'
            node.procedures.append(new_node)
    return tree


def add_missing_destructors(tree):
    for node in ft.walk(tree):
        if not isinstance(node, ft.Type):
            continue
        for child in ft.iter_child_nodes(node):
            if 'destructor' in child.attributes:
                log.info('found destructor %s', child.name)
                break
        else:

            log.info('adding missing destructor for %s', node.name)
            new_node = ft.Subroutine('%s_finalise' % node.name,
                                     node.filename,
                                     ['Automatically generated destructor for %s' % node.name],
                                     node.lineno,
                                     [ft.Argument(name='this',  # self.prefix + 'this' would probably be safer
                                                  filename=node.filename,
                                                  doc=['Object to be destructed'],
                                                  lineno=node.lineno,
                                                  attributes=['intent(inout)'],
                                                  type='type(%s)' % node.name)],
                                     node.uses,
                                     ['destructor', 'skip_call'],
                                     mod_name=node.mod_name,
                                     type_name=node.name)
            new_node.method_name = '__del__'
            node.procedures.append(new_node)
    return tree


class FunctionToSubroutineConverter(ft.FortranTransformer):
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

        new_node = ft.Subroutine(node.name,
                                 node.filename,
                                 node.doc,
                                 node.lineno,
                                 arguments,
                                 node.uses,
                                 node.attributes,
                                 mod_name=node.mod_name)
        if hasattr(node, 'call_name'):
            new_node.call_name = node.call_name
        if hasattr(node, 'type'):
            new_node.type = node.type
        new_node.orig_name = node.orig_name
        new_node.orig_node = node  # keep a reference to the original node
        return new_node


class IntentOutToReturnValues(ft.FortranTransformer):
    """
    Convert all Subroutine and Function intent(out) arguments to return values
    """

    def visit_Procedure(self, node):
        if 'constructor' in node.attributes:
            node.arguments[0].attributes = set_intent(node.arguments[0].attributes,
                                                      'intent(out)')

        ret_val = []
        ret_val_doc = None
        if isinstance(node, ft.Function) and node.ret_val is not None:
            ret_val.append(node.ret_val)
            if node.ret_val_doc is not None:
                ret_val_doc = node.ret_val_doc

        arguments = []
        for arg in node.arguments:
            if 'intent(out)' in arg.attributes:
                ret_val.append(arg)
            else:
                arguments.append(arg)
        if ret_val == []:
            new_node = node  # no changes needed
        else:
            new_node = ft.Function(node.name,
                                   node.filename,
                                   node.doc,
                                   node.lineno,
                                   arguments,
                                   node.uses,
                                   node.attributes,
                                   ret_val,
                                   ret_val_doc,
                                   mod_name=node.mod_name,
                                   type_name=node.type_name)
            new_node.orig_node = node
            if hasattr(node, 'method_name'):
                new_node.method_name = node.method_name
        return new_node


class RenameReservedWords(ft.FortranVisitor):
    def __init__(self, types, name_map=None):
        self.types = types
        self.name_map = {}
        if name_map is not None:
            self.name_map.update(name_map)

        # rename Python keywords by appending an underscore
        import keyword
        self.name_map.update(dict((key, key + '_') for key in keyword.kwlist))

        # apply same renaming as f2py
        import numpy.f2py.crackfortran
        self.name_map.update(numpy.f2py.crackfortran.badnames)

        # avoid clashes with C intrinsic functions
        self.name_map['inverse'] = 'inverse_'

        # remove some of these which are not Python reserved words
        del self.name_map['stdout']
        del self.name_map['stderr']
        del self.name_map['stdin']

    def visit_Argument(self, node):
        if not hasattr(node, 'orig_name'):
            node.orig_name = node.name
        node.name = self.name_map.get(node.name, node.name)
        if isinstance(node, ft.Argument):
            # replace names in dimension attribute expressions
            for (old_name, new_name) in self.name_map.items():
                new_attribs = []
                for attrib in node.attributes:
                    if attrib.startswith('dimension('):
                        new_attribs.append(attrib.replace(old_name, new_name))
                    else:
                        new_attribs.append(attrib)
                node.attributes = new_attribs
        return self.generic_visit(node)

    visit_Procedure = visit_Argument
    visit_Element = visit_Argument
    visit_Module = visit_Argument
    visit_Type = visit_Argument


class RenameArgumentsPython(ft.FortranVisitor):
    def __init__(self, types):
        self.types = types

    def visit_Procedure(self, node):
        if hasattr(node, 'method_name'):
            if 'constructor' in node.attributes:
                node.ret_val[0].py_name = 'self'
            elif len(node.arguments) >= 1 and node.arguments[0].type in self.types:
                node.arguments[0].py_name = 'self'
        elif hasattr(node, 'attributes') and 'callback' in node.attributes:
            self.visit_Argument(node)
        return self.generic_visit(node)

    def visit_Argument(self, node):
        if not hasattr(node, 'py_name'):
            node.py_name = node.name
        if ft.is_derived_type(node.type):
            node.py_value = node.py_name + '._handle'
        else:
            node.py_value = node.py_name
        return node


class RenameInterfacesPython(ft.FortranVisitor):
    def visit_Interface(self, node):
        for proc in node.procedures:
            if hasattr(proc, 'method_name'):
                proc.method_name = '_' + proc.method_name
            else:
                proc.method_name = '_' + proc.name
        node.method_name = node.name
        if node.name == 'assignment(=)':
            node.method_name = 'assignment'
        elif node.name == 'operator(+)':
            node.method_name = '__add__'
        elif node.name == 'operator(-)':
            node.method_name = '__sub__'
        elif node.name == 'operator(*)':
            node.method_name = '__mul__'
        elif node.method_name == 'operator(/)':
            node.method_name = '__div__'
        elif '(' in node.name:
            raise RuntimeError("unsupported operator overload '%s'" % node.name)
        return node


class ReorderOptionalArgumentsPython(ft.FortranVisitor):
    """
    Move optional arguments after non-optional arguments:
    in Fortran they can come in any order, but in Python
    optional arguments must come at the end of argument list.
    """

    def visit_Procedure(self, node):
        non_optional = [arg for arg in node.arguments if 'optional' not in arg.attributes]
        optional = [arg for arg in node.arguments if 'optional' in arg.attributes]
        node.arguments = non_optional + optional
        return node


class OnlyAndSkip(ft.FortranTransformer):
    """
    This class does the job of removing nodes from the tree
    which are not necessary to write wrappers for (given user-supplied
    values for only and skip).

    Currently it takes a list of subroutines and a list of modules to write
    wrappers for. If empty, it does all of them.
    """

    def __init__(self, kept_subs, kept_mods):
        self.kept_subs = kept_subs
        self.kept_mods = kept_mods

    def visit_Procedure(self, node):
        if len(self.kept_subs) > 0:
            if node not in self.kept_subs:
                return None
        return self.generic_visit(node)

    def visit_Module(self, node):
        if len(self.kept_mods) > 0:
            if node not in self.kept_mods:
                return None
        return self.generic_visit(node)


class NormaliseTypes(ft.FortranVisitor):
    """
    Convert all type names to standard form and resolve kind names
    """

    def __init__(self, kind_map):
        self.kind_map = kind_map

    def visit_Declaration(self, node):
        node.type = ft.normalise_type(node.type, self.kind_map)
        return self.generic_visit(node)

    visit_Argument = visit_Declaration


class SetInterfaceProcedureCallNames(ft.FortranVisitor):
    """
    Set call names of procedures within overloaded interfaces to the name of the interface
    """

    def visit_Interface(self, node):
        for proc in node.procedures:
            log.info('setting call_name of %s to %s' % (proc.name, node.name))
            proc.call_name = node.name
        return node


class ResolveInterfacePrototypes(ft.FortranTransformer):
    """
    Replaces prototypes in interface declarations with referenced procedure
    """
    def visit_Module(self, node):

        # Attempt 0:
        # Insert procedure at first interface reference. Does not support
        # procedures being reused in multiple interfaces. This was how the
        # original resolution logic was implemented when resolution occurred in
        # the parser. Technically this is quadratic complexity, but the number
        # of interface prototypes is generally small...
        def inject_procedure(interfaces, procedure):
            for iface in interfaces:
                for i, p in enumerate(iface.procedures):
                    if procedure.name == p.name:
                        log.debug("Procedure %s moved to interface %s", procedure.name, iface.name)
                        iface.procedures[i] = procedure # Replace the prototype
                        return True
            log.debug(f"Procedure %s is not used in any interface", procedure.name)
            return False

        unused = []
        for mp in node.procedures:
            if not inject_procedure(node.interfaces, mp):
                unused.append(mp)
        node.procedures = unused
        return node

        # Attempt 1:
        # Insert procedures at first reference. Elegant and equivalent to Option 0,
        # but causes issues because it throws an error if procedure is referenced by
        # multiple interfaces (Option 0 just ignores unresolved prototypes)
        #procedure_map = { p.name:p for p in node.procedures }
        #for int in node.interfaces:
        #    int.procedures = [ procedure_map.pop(p.name) for p in int.procedures ]
        #node.procedures = list(procedure_map.values())

        # Option 2:
        # Support procedure reuse across multiple interfaces by inserting the
        # Procedure node once per interface node. This causes problems in
        # fortran code gen b/c identically named wrappers will be generated for
        # each interface, causing a name clash. This could be fixed in code gen
        # by adding the interface name to the wrapper function name.
        #procedure_map = { p.name:p for p in node.procedures }
        #unused = set(procedure_map.keys())
        #for int in node.interfaces:
        #    iprocs = { p.name for p in int.procedures }
        #    unused -= iprocs # Can't eagerly remove b/c may be in multiple interfaces
        #    int.procedures = [ procedure_map[p] for p in iprocs ]
        #node.procedures = [ procedure_map[p] for p in unused ]
        #return node


class ResolveBindingPrototypes(ft.FortranTransformer):
    """
    Replaces prototypes in type binding declarations with referenced procedure.

    FIXME: Fortran allows module procedures to be bound to more than one type
           procedure, i.e. x%fun1() and x%fun2() can both bind to module
           procedure x_fun(). The approach below only support 1-to-1 mappings.
    """
    def visit_Module(self, node):
        procedure_map = { p.name:p for p in node.procedures }
        for type in node.types:

            # Pass 1: Associate module procedures with specific bindings
            for ib, binding in enumerate(type.bindings):
                if binding.type == 'generic':
                    continue
                proto = binding.procedures[0]  # Only generics have multiple procedures
                proc = procedure_map.pop(proto.name)
                log.debug('Creating method for %s from procedure %s.', type.name, proc.name)
                proc.type_name = type.name
                proc.method_name = binding.name
                proc.attributes.append('method')
                if binding.type == 'final':
                    log.debug('Marking method %s as destructor for %s', proc.method_name, type.name)
                    proc.attributes.append('destructor')
                    type.bindings[ib].attributes.append('destructor')
                binding.procedures = [proc]

            # Pass 2: Consolidate specific bindings into generic bindings, if needed
            binding_map = { b.name:b for b in type.bindings }
            for binding in type.bindings:
                if binding.type != 'generic':
                    continue
                # For generics, prototypes name specific bindings
                binding.procedures = [ binding_map.pop(p.name).procedures[0] for p in binding.procedures ]
            type.bindings = list(binding_map.values())

        node.procedures = list(procedure_map.values())
        return node


class BindConstructorInterfaces(ft.FortranTransformer):
    """
    Moves interfaces named after a type into that type and marks as constructor.

    The Fortran idiom for defining custom constructors is to create an interface
    with the same name as the type which references one or more free functions
    that return initialized instances of the type. This transformer locates such
    interfaces, moves them into the type, and marks them as constructors.
    """
    def visit_Module(self, node):
        interface_map = { i.name:i for i in node.interfaces }
        for type in node.types:
            if type.name not in interface_map:
                continue
            interface = interface_map.pop(type.name)
            log.debug('Move interface %s into type %s and mark constructor', interface.name, type.name)
            interface.attributes.append('constructor')
            for p in interface.procedures:
                p.type_name = type.name
                p.attributes.append('constructor')
            type.interfaces.append(interface)
        node.interfaces = list(interface_map.values())
        return node


def transform_to_generic_wrapper(tree, types, callbacks, constructors,
                                 destructors, short_names, init_lines,
                                 kept_subs, kept_mods, argument_name_map,
                                 move_methods, shorten_routine_names,
                                 modules_for_type, remove_optional_arguments,
                                 force_public=None):
    """
    Apply a number of rules to *tree* to make it suitable for passing to
    a F90 and Python wrapper generators. Transformations performed are:

     * Removal of procedures and modules not provided by the user
     * Removal of private symbols
     * Removal of unwrappable routines and optional arguments
     * Addition of missing constructor and destructor wrappers
     * Conversion of all functions to subroutines
     * Updating call names of procedures within interfaces
     * Update of subroutine uses clauses
    """

    tree = ResolveInterfacePrototypes().visit(tree)
    tree = ResolveBindingPrototypes().visit(tree)
    tree = BindConstructorInterfaces().visit(tree)
    tree = OnlyAndSkip(kept_subs, kept_mods).visit(tree)
    tree = remove_private_symbols(tree, force_public)
    tree = UnwrappablesRemover(callbacks, types, constructors,
                               destructors, remove_optional_arguments).visit(tree)
    tree = fix_subroutine_uses_clauses(tree, types)
    tree = MethodFinder(types, constructors, destructors, short_names,
                        move_methods, shorten_routine_names, modules_for_type).visit(tree)
    SetInterfaceProcedureCallNames().visit(tree)
    tree = collapse_single_interfaces(tree)
    tree = fix_subroutine_uses_clauses(tree, types) # do it again, to fix interfaces
    create_super_types(tree, types)  # This must happen before fix_subroutine_type_arrays
    fix_subroutine_type_arrays(tree, types)  # This must happen after fix_subroutine_uses_clauses, to avoid using the
    # super-types that do not exist and use the regular type, which is used in the declaration of the super-type
    tree = fix_element_uses_clauses(tree, types)
    tree = add_missing_constructors(tree)
    tree = add_missing_destructors(tree)
    tree = ConstructorExcessToClassMethod().visit(tree) # dealing with cases where multiple constructors were allocated
    tree = convert_array_intent_out_to_intent_inout(tree)
    RenameReservedWords(types, argument_name_map).visit(tree)
    return tree


def transform_to_f90_wrapper(tree, types, callbacks, constructors,
                             destructors, short_names, init_lines,
                             string_lengths, default_string_length,
                             sizeof_fortran_t, kind_map):
    """
    Additional Fortran-specific transformations:
     * Conversion of derived type arguments to opaque integer arrays
       via Fortran transfer() intrinsic.
    * Normalise type declarations
    """
    FunctionToSubroutineConverter().visit(tree)
    tree = convert_derived_type_arguments(tree, init_lines, sizeof_fortran_t)
    StringLengthConverter(string_lengths, default_string_length).visit(tree)
    ArrayDimensionConverter().visit(tree)
    NormaliseTypes(kind_map).visit(tree)
    return tree


def transform_to_py_wrapper(tree, types):
    """
    Additional Python-specific transformations:
      * Convert intent(out) arguments to additional return values
      * Rename arguments (e.g. this -> self)
      * Prefix procedure names within interfaces with an underscore
    """
    IntentOutToReturnValues().visit(tree)
    RenameArgumentsPython(types).visit(tree)
    ReorderOptionalArgumentsPython().visit(tree)
    RenameInterfacesPython().visit(tree)
    return tree


def find_referenced_modules(mods, tree):
    """
    Given a set of modules in a parse tree, find any modules (recursively)
    used by these.

    Parameters
    ----------
    mods : set
        initial modules to search, must be included in the tree.

    tree : `fortran.Root()` object.
        the full fortran parse tree from which the mods have been taken.

    Returns
    -------
    all_mods : set
        Module() objects which are recursively used by the given modules.
    """
    new_mods = copy.copy(mods)
    while new_mods != set():
        temp = list(new_mods)
        for m in temp:
            for m2 in m.uses:
                for m3 in ft.walk_modules(tree):
                    if m3.name == m2:
                        new_mods.add(m3)
        new_mods -= mods
        mods |= new_mods

    return mods


def find_referenced_types(mods, tree):
    """
    Given a set of modules in a parse tree, find any types either defined in
    or referenced by the module, recursively.

    Parameters
    ----------
    mods : set
        initial modules to search, must be included in the tree.

    tree : the full fortran parse tree from which the mods have been taken.
    tree : `fortran.Root` object.
        the full fortran parse tree from which the mods have been taken.

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
            if ft.is_derived_type(el.type):
                for mod2 in ft.walk_modules(tree):
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
                if ft.is_derived_type(el.type):  # a referenced type, need to find def
                    for mod2 in ft.walk_modules(tree):
                        for mt in mod2.types:
                            if mt.name in el.type:
                                new_set.add(mt)
        # take out all the original types from new_set
        new_set -= kept_types
        # update the kept_types with new ones
        kept_types |= new_set

    return kept_types


def create_super_types(tree, types):
    # YANN: Gather all the dimensions of arrays of type in arguments and module variables
    # Add here any other elements where arrays of derived types may appear
    append_type_dimension(tree, types)
    # Then create a super-type for each type for each of those dimension, inside the module where
    # the type is declared in the first place.
    modules_indexes = dict(
        (mod.name, i) for (i, mod) in enumerate(tree.modules))  # name to index map, compatible python 2.6
    containers = []
    for ty in types.values():
        for dimensions_attribute in ty.super_types_dimensions:
            # each type might have many "dimension" attributes since "append_type_dimension"
            dimensions = ArrayDimensionConverter.split_dimensions(dimensions_attribute)
            if len(dimensions) == 1:  # at this point, only 1D arrays are supported
                d = dimensions[0]
                if str(d) == ':':
                    continue  # at this point, only fixed-length arrays are supported
                # populate the super type with the array of types
                el = ft.Element(name='items', attributes=[dimensions_attribute], type='type(' + ty.name + ')')
                name = ty.name + '_x' + str(d) + '_array'
                # populate the tree with the super-type
                if name not in (t.name for t in tree.modules[modules_indexes[ty.mod_name]].types):
                    super_type = ft.Type(name=name, filename=ty.filename, lineno=ty.lineno,
                                         doc=['super-type',
                                              'Automatically generated to handle derived type arrays as a new derived type'],
                                         elements=[el], mod_name=ty.mod_name)
                    # uses clauses from the base type
                    # super_type.uses = ty.uses  # this causes unwanted growth of the normal type "uses" when we add parameters to the super-type in the next step
                    super_type.uses = set([(ty.mod_name, (ty.name,))])
                    # uses clause if the dimension is a parameter (which is handled through a n=shape(array) hidden argument in the case of regular arrays)
                    param = extract_dimensions_parameters(d, tree)
                    if param:
                        super_type.uses.add((param[0], (param[1],)))

                    tree.modules[modules_indexes[ty.mod_name]].types.append(super_type)
                    containers.append(tree.modules[modules_indexes[ty.mod_name]].types[-1])

    for ty in containers:
        types['type(%s)' % ty.name] = types[ty.name] = ty


def append_type_dimension(tree, types):
    # YANN: save the dimensions of all the type arrays in the base-type, in order to create a super-type for each of them afterwards
    from itertools import chain
    for proc in chain(tree.procedures, *(mod.procedures for mod in tree.modules)):
        for arg in proc.arguments:
            if arg.type in types:
                types[arg.type].super_types_dimensions.update(attrib for attrib in arg.attributes if
                                                            attrib.startswith('dimension'))


def fix_subroutine_type_arrays(tree, types):
    # YANN: replace dimension(x) :: type() arguments of routines by scalar super-types
    from itertools import chain
    # For each top-level procedure and module procedures:
    for proc in chain(tree.procedures, *(mod.procedures for mod in tree.modules)):
        for arg in proc.arguments:
            dimensions_attribute = [attr for attr in arg.attributes if attr.startswith('dimension')]
            if ft.is_derived_type(arg.type) and len(dimensions_attribute) == 1:
                # an argument should only have 0 or 1 "dimension" attributes
                # If the argument is an 1D-array of types, convert it to super-type:
                d = ArrayDimensionConverter.split_dimensions(dimensions_attribute[0])[0]
                if str(d) == ':':
                    continue
                # change the type to super-type
                arg.type = arg.type[:-1] + '_x' + str(d) + '_array)'
                # if the dimension is a parameter somewhere, add it to the uses clauses
                param = extract_dimensions_parameters(d, tree)
                if param:
                    proc.uses.add((param[0], (param[1],)))
                # ... then remove the dimension, since we now use a scalar super-type ...
                arg.attributes = [attr for attr in arg.attributes if not attr.startswith('dimension')]
                # ... and brand it for the final call
                arg.doc.append('super-type')


def extract_dimensions_parameters(d, tree):
    # YANN: returns (module, parameter) if there is a parameter matching the dimension
    if not d.isdigit():
        # then: look for the dimension in the parameters
        for mod in tree.modules:
            for el in mod.elements:
                if d == el.name and "parameter" in el.attributes:
                    return (mod.name, el.name)

def shorten_long_name(name):
    fortran_max_name_length = 63
    hash_length = 4
    if len(name) > fortran_max_name_length:
        name_hash = hashlib.md5(name.lower().encode('utf-8')).hexdigest()
        shorter_name = name[:fortran_max_name_length-hash_length] + name_hash[:hash_length]
        log.info('Renaming "%s" to "%s" to comply with Fortran 2003'%(name, shorter_name))
        return shorter_name
    return name
