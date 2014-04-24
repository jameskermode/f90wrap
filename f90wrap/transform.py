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
import re

import numpy as np

from f90wrap.fortran import (Fortran, Root, Program, Module, Procedure, Subroutine, Function,
                             Declaration, Element, Argument, Type, Interface,
                             FortranVisitor, FortranTransformer,
                             walk, walk_procedures, walk_modules,
                             iter_child_nodes, strip_type)


class AccessUpdater(FortranTransformer):
    """Visit module contents and update public_symbols and
       private_symbols lists to be consistent with (i) default module
       access; (ii) public and private statements at module level;
       (iii) public and private attibutes."""

    def __init__(self):
        self.mod = None

    def visit_Module(self, mod):
        # keep track of the current module
        self.mod = mod
        self.generic_visit(mod)
        self.mod = None

    def visit(self, node):
        if self.mod is None:
            return self.generic_visit(node)

        if self.mod.default_access == 'public':
            if ('private' not in getattr(node, 'attributes', {}) and
                   node.name not in self.mod.private_symbols):

                # symbol should be marked as public if it's not already
                if node.name not in self.mod.public_symbols:
                    logging.debug('marking public symbol ' + node.name)
                    self.mod.public_symbols.append(node.name)
            else:
                # symbol should be marked as private if it's not already
                if node.name not in self.mod.private_symbols:
                    logging.debug('marking private symbol ' + node.name)
                    self.mod.private_symbols.append(node.name)

        elif self.mod.default_access == 'private':
            if ('public' not in getattr(node, 'attributes', {}) and
                   node.name not in self.mod.public_symbols):

                # symbol should be marked as private if it's not already
                if node.name not in self.mod.private_symbols:
                    logging.debug('marking private symbol ' + node.name)
                    self.mod.private_symbols.append(node.name)
            else:
                # symbol should be marked as public if it's not already
                if node.name not in self.mod.public_symbols:
                    logging.debug('marking public symbol ' + node.name)
                    self.mod.public_symbols.append(node.name)

        else:
            raise ValueError('bad default access %s for module %s' %
                               (self.mod.default_access, self.mod.name))

        return node  # no need to recurse further


class PrivateSymbolsRemover(FortranTransformer):
    """
    Transform a tree by removing private symbols.
    """

    def __init__(self):
        self.mod = None

    def visit_Module(self, mod):
        # keep track of the current module
        self.mod = mod
        self.generic_visit(mod)
        self.mod = None

    def visit(self, node):
        if self.mod is None:
            return self.generic_visit(node)

        if node.name in self.mod.private_symbols:
            logging.debug('removing private symbol ' + node.name)
            return None
        else:
            return node

def remove_private_symbols(node):
    """
    Walk the tree starting at *node*, removing all private symbols.

    This funciton first applies the AccessUpdater transformer to
    ensure module *public_symbols* and *private_symbols* are up to
    date with *default_access* and individual `public` and `private`
    attributes.
    """

    node = AccessUpdater().visit(node)
    node = PrivateSymbolsRemover().visit(node)
    return node

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
        """
        Remove unwrappeble elements inside derived types
        """
        if node.name not in self.types:
            logging.debug('removing type %s' % node.name)
            return None
        else:
            elements = []
            for element in node.elements:
                # Get the number of dimensions of the element (if any)
                dims = filter(lambda x: x.startswith('dimension'), element.attributes)
                # Skip this if the type is not do-able
                if 'pointer' in element.attributes and dims != []:
                    continue
                if element.type.lower() == 'type(c_ptr)':
                    continue
                elements.append(element)
            node.elements = elements
            return node


    def visit_Module(self, node):
        """
        Remove unwrappable elements inside modules.

        As above, but also includes derived type elements from modules
        that do not have the "target" attribute
        """
        elements = []
        for element in node.elements:
            # Get the number of dimensions of the element (if any)
            dims = filter(lambda x: x.startswith('dimension'), element.attributes)
            # Skip this if the type is not do-able
            if 'pointer' in element.attributes and dims != []:
                continue
            if element.type.lower() == 'type(c_ptr)':
                continue
            if element.type.startswith('type(') and 'target' not in element.attributes:
                continue
            elements.append(element)
        node.elements = elements
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

            typename = strip_type(arg.type)
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
                    logging.info('found constructor %s' % child.name)
                    break
        else:
            logging.info('adding missing constructor for %s' % node.name)
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
                    logging.info('found destructor %s' % child.name)
                    break
        else:
            logging.info('adding missing destructor for %s' % node.name)
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


class FunctionToSubroutineConverter(FortranTransformer):
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

class IntentOutToReturnValues(FortranTransformer):
    """
    Convert all Subroutine and Function intent(out) arguments to return values
    """

    def visit_Procedure(self, node):
        if 'constructor' in node.attributes:
            node.arguments[0].attributes = set_intent(node.arguments[0].attributes,
                                                      'intent(out)')

        ret_val = []
        ret_val_doc = None
        if isinstance(node, Function) and node.ret_val is not None:
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
            new_node = Function(node.name,
                                node.filename,
                                node.doc,
                                node.lineno,
                                arguments,
                                node.uses,
                                node.attributes,
                                ret_val,
                                ret_val_doc)
        new_node.orig_node = node
        return new_node

class RenameArguments(FortranVisitor):
    def __init__(self, name_map=None):
        if name_map is None:
            name_map = {'this': 'self'}
        self.name_map = name_map

    def visit_Argument(self, node):
        node.orig_name = node.name
        node.name = self.name_map.get(node.name, node.name)
        if node.type.startswith('type('):
            node.value = node.name + '._handle'
        else:
            node.value = node.name
        return node


class OnlyAndSkip(FortranTransformer):
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


def transform_to_generic_wrapper(tree, types, kinds, callbacks, constructors,
                                 destructors, short_names, init_lines,
                                 only_subs, only_mods):
    """
    Apply a number of rules to *tree* to make it suitable for passing to
    a F90 and Python wrapper generators. Transformations performed are:

     * Removal of procedures and modules not provided by the user
     * Removal of private symbols
     * Removal of unwrappable routines and optional arguments
     * Addition of missing constructor and destructor wrappers
     * Conversion of all functions to subroutines
     * Update of subroutine uses clauses
    """
    tree = OnlyAndSkip(only_subs, only_mods).visit(tree)
    tree = remove_private_symbols(tree)
    tree = UnwrappablesRemover(callbacks, types, constructors, destructors).visit(tree)
    tree = MethodFinder(types, constructors, destructors, short_names).visit(tree)
    tree = collapse_single_interfaces(tree)
    tree = add_missing_constructors(tree)
    tree = add_missing_destructors(tree)
    return tree

def transform_to_f90_wrapper(tree, types, kinds, callbacks, constructors,
                             destructors, short_names, init_lines,
                             string_lengths, default_string_length,
                             sizeof_fortran_t):
    """
    Additional Fortran-specific transformations:
     * Conversion of derived type arguments to opaque integer arrays
       via Fortran transfer() intrinsic.
    """
    FunctionToSubroutineConverter().visit(tree)
    tree = fix_subroutine_uses_clauses(tree, types, kinds)
    tree = convert_derived_type_arguments(tree, init_lines, sizeof_fortran_t)
    StringLengthConverter(string_lengths, default_string_length).visit(tree)
    ArrayDimensionConverter().visit(tree)
    return tree

def transform_to_py_wrapper(tree, argument_name_map=None):
    """
    Additional Python-specific transformations:
      * Convert intent(out) arguments to additional return values
      * Rename arguments (e.g. this -> self)
    """
    IntentOutToReturnValues().visit(tree)
    RenameArguments(argument_name_map).visit(tree)
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
                for m3 in walk_modules(tree):
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
