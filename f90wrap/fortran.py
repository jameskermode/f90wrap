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
"""
This module defines a series of classes which inherit an abstract base class.
Each represents a node in a Fortran parse tree -- Modules, Subroutines,
Arguments etc. A Fortran parse tree will contain only these classes as nodes.
"""

from __future__ import print_function
import logging
import re

import numpy as np

log = logging.getLogger(__name__)


def _rep_des(doc, string):
    """
    Replaces the description line of a documentation with `string`
    """
    doc_desc = doc.split("\n")[0] or doc.split("\n")[1]
    doc_desc = doc_desc.lstrip()
    doc = doc.replace(doc_desc, string)
    return doc

class Fortran(object):
    """
    Abstract base class for all nodes in Fortran parser tree.

    Parameters
    ----------
    name : `str`, default ``""``
        Name of the node

    filename : `str`, default ``""``
        Name of the file in which node is defined

    doc : `list` of `str`, default ``None``
        Documentation found in the node

    lineno : `int`, default ``0``.
        Line number at which the node begins.
    """

    _fields = []

    def __init__(self, name='', filename='', doc=None,
                 lineno=0, doxygen=''):
        self.name = name
        self.filename = filename
        if doc is None:
            doc = []
        self.doc = doc
        self.doxygen = doxygen
        self.lineno = lineno

    def __repr__(self):
        return '%s(name=%s)' % (self.__class__.__name__, self.name)

    def __eq__(self, other):
        # FIXME: could not we use getattr here to simplify all cases?
        if other is None: return False
        attrs = [el for el in self.__dict__ if not el.startswith("_")]
        ret = True
        if type(other) != type(self):
            return False
        for a in attrs:
            try:
                ret = ret and getattr(self, a) == getattr(other, a)
            except:
                return False
            if not ret:
                return False
        return True

    def __neq__(self, other):
        return not self.__eq__(other)


class Root(Fortran):
    """
    programs : `list` of `fortran.Program`, default ``None``
        A list of Programs within the parse tree.

    modules : `list` of `fortran.Module`, default ``None``
        A list of modules within the parse tree

    procedures : `list` of `fortran.Procedure`, default ``None``
        A list of top-level procedures within the parse tree.
    """
    __doc__ = _rep_des(Fortran.__doc__, "The Root node of a Fortan parse tree") + __doc__
    _fields = ['programs', 'modules', 'procedures']

    def __init__(self, name='', filename='', doc=None, lineno=0,
                 programs=None, modules=None, procedures=None):
        Fortran.__init__(self, name, filename, doc, lineno)
        if programs is None:
            programs = []
        self.programs = programs
        if modules is None:
            modules = []
        self.modules = modules
        if procedures is None:
            procedures = []
        self.procedures = procedures


class Program(Fortran):
    """
    procedures : list of :class:`fortran.Procedure` , default ``None``
        A list of procedures within the program's scope.
    """
    __doc__ = _rep_des(Fortran.__doc__, "Class to represent a Fortran main program.") + __doc__
    _fields = ['procedures']

    def __init__(self, name='', filename='', doc=None, lineno=0,
                 procedures=None, uses=None):
        Fortran.__init__(self, name, filename, doc, lineno)
        if procedures is None:
            procedures = []
        self.procedures = procedures
        if uses is None:
            uses = []
        self.uses = uses


class Module(Fortran):
    """
    types : list of :class:`fortran.Type` , default ``None``
        Derived-types defined in the module

    elements : list of :class:`fortran.Element` , default ``None``
        Module-level variables in the module

    procedures : list of :class:`fortran.Procedure` , default ``None``
        A list of procedures defined in the module

    interfaces : list of :class:`fortran.Interface` , default ``None``
        A list of interfaces defined in the module

    uses : list of `str` or `tuple` , default ``None``
        A list of modules that this module uses. If the entry is a tuple, it
        should be in the form (uses,only,[only,..]), where the `only` entries
        are subroutines/elements in the used module.

    default_access : `str`, default ``"public"``
        The default access to the module (public or private)

    public_symbols : list of `str` , default ``None``
        The symbols within the module that are public

    private_symbols : list of `str` , default ``None``
        The symbols within the module that are private
    """
    __doc__ = _rep_des(Fortran.__doc__, "Represents a Fortran module.") + __doc__
    _fields = ['types', 'elements', 'procedures', 'interfaces', 'uses']

    def __init__(self, name='', filename='', doc=None, lineno=0,
                 types=None, elements=None, procedures=None,
                 interfaces=None, uses=None, default_access='public',
                 public_symbols=None, private_symbols=None):
        Fortran.__init__(self, name, filename, doc, lineno)
        if types is None:
            types = []
        self.types = types
        if elements is None:
            elements = []
        self.elements = elements
        if procedures is None:
            procedures = []
        self.procedures = procedures
        if interfaces is None:
            interfaces = []
        self.interfaces = interfaces
        if uses is None:
            uses = []
        self.uses = uses
        self.default_access = default_access
        if public_symbols is None:
            public_symbols = []
        self.public_symbols = public_symbols
        if private_symbols is None:
            private_symbols = []
        self.private_symbols = private_symbols

    # Required for the Module object to be hashable so one can create sets of Modules
    # So this function should return a unique imprint of the object
    # I guess the filename + the module name should be unique enough ?
    # Also, hash requires an integer, so we convert the string to integers with the
    # same number of digits to ensure one-to-one conversion.
    # This is maybe unnecessarily long ?
    def __hash__(self):
        return int(''.join(str(ord(x)).zfill(3) for x in self.filename + self.name))


class Procedure(Fortran):
    """
    arguments : list of :class:`fortran.Argument`
        A list of arguments to the procedure

    uses : list of `str` or `tuple` , default ``None``
        A list of modules that this procedure uses. If the entry is a tuple, it
        should be in the form (uses,only,[only,..]), where the `only` entries
        are subroutines/elements in the used module.

    attributes : list of `str`, default ``None``
        Attributes of the procedure

    mod_name : `str` , default ``None``
        The name of the module in which the procedure is found, if any.

    type_name : `str` , default ``None``
        The name of the type in which the procedure is defined, if any.
    """
    __doc__ = _rep_des(Fortran.__doc__, "Represents a Fortran Function or Subroutine.") + __doc__
    _fields = ['arguments']

    def __init__(self, name='', filename='', doc=None, lineno=0,
                 arguments=None, uses=None, attributes=None,
                 mod_name=None, type_name=None):
        Fortran.__init__(self, name, filename, doc, lineno)
        if arguments is None: arguments = []
        self.arguments = arguments
        if uses is None: uses = []
        self.uses = uses
        if attributes is None: attributes = []
        self.attributes = attributes
        self.mod_name = mod_name
        self.type_name = type_name

class Subroutine(Procedure):
    __doc__ = _rep_des(Procedure.__doc__, "Represents a Fortran Subroutine.")
    pass

class Function(Procedure):
    """
    ret_val : :class:`fortran.Argument`
        The argument which is the returned value

    ret_val_doc : `str`
        The documentation of the returned value
    """
    __doc__ = _rep_des(Procedure.__doc__, "Represents a Fortran Function.") + __doc__
    _fields = ['arguments', 'ret_val']

    def __init__(self, name='', filename='', doc=None, lineno=0,
                 arguments=None, uses=None, attributes=None,
                 ret_val=None, ret_val_doc=None, mod_name=None, type_name=None):
        Procedure.__init__(self, name, filename, doc,
                           lineno, arguments, uses, attributes,
                           mod_name, type_name)
        if ret_val is None:
            ret_val = Argument()
        self.ret_val = ret_val
        self.ret_val_doc = ret_val_doc


class Prototype(Fortran):
    __doc__ = _rep_des(Fortran.__doc__, "Represents a Fortran Prototype.")
    pass

class Declaration(Fortran):
    """
    type : `str` , default ``""``
        The type of the declaration

    attributes : list of `str` , default ``None``
        A list of attributes defined in the declaration (eg. intent(in), allocatable)

    value : `str`
        A value given to the variable upon definition
        (eg. value=8 in ``"integer :: x = 8"``
    """
    __doc__ = _rep_des(Fortran.__doc__, "Base class representing a declaration statement") + __doc__
    def __init__(self, name='', filename='', doc=None, lineno=0,
                 attributes=None, type='', value='', doxygen=''):
        Fortran.__init__(self, name, filename, doc, lineno, doxygen=doxygen)
        if attributes is None: attributes = []
        self.attributes = attributes
        self.type = type
        self.value = value


class Element(Declaration):
    __doc__ = _rep_des(Declaration.__doc__, "Represents a Module or Derived-Type Element.")
    pass

class Argument(Declaration):
    __doc__ = _rep_des(Declaration.__doc__, "Represents a Procedure Argument.")
    pass

class Type(Fortran):
    """
    elements : list of :class:`fortran.Element`
        Variables within the type

    procedures : list of :class:`fortran.Procedure`
        Procedures defined with the type.
    """
    __doc__ = _rep_des(Fortran.__doc__, "Represents a Fortran Derived-type.") + __doc__
    _fields = ['elements', 'procedures', 'bindings', 'interfaces']

    def __init__(self, name='', filename='', doc=None,
                 lineno=0, elements=None, procedures=None, bindings=None, interfaces=None,
                 mod_name=None):
        Fortran.__init__(self, name, filename, doc, lineno)
        self.elements = elements if elements else []
        self.procedures = procedures if procedures else []
        self.bindings = bindings if bindings else []
        self.interfaces = interfaces if interfaces else []
        self.mod_name = mod_name
        self.super_types_dimensions = set()


class Interface(Fortran):
    """
    procedures : list of :class:`fortran.Procedure`
        The procedures listed in the interface.

    mod_name : `str` , default ``None``
        The name of the module in which the interface is found, if any.

    type_name : `str` , default ``None``
        The name of the type in which the interface is defined, if any.
    """
    __doc__ = _rep_des(Fortran.__doc__, "Represents a Fortran Interface.") + __doc__
    _fields = ['procedures']

    def __init__(self, name='', filename='', doc=None,
                 lineno=0, procedures=None, attributes=None, mod_name=None, type_name=None):
        Fortran.__init__(self, name, filename, doc, lineno)
        self.procedures = procedures if procedures else []
        self.attributes = attributes if attributes else []
        self.mod_name = mod_name
        self.type_name = type_name

class Binding(Fortran):
    """
    type : `str`, default ``None``
        The type of bound procedures: ['procedure', 'generic', 'final']

    attributes : list of `str`, default ``[]``
        Attributes of the procedure

    procedures : list of :class:`fortran.Procedure`, default ``[]``
        The procedures listed in the binding.

    mod_name : `str` , default ``None``
        The name of the module in which the interface is found, if any.

    type_name : `str` , default ``None``
        The name of the type in which the interface is defined, if any.
    """
    __doc__ = _rep_des(Fortran.__doc__, "Represents a Derived Type procedure binding.") + __doc__
    _fields = ['procedures']

    __doc__ = _rep_des(Fortran.__doc__, "Represents a type procedure binding.")
    def __init__(self, name='', filename='', doc=None, lineno=0,
                 type=None, attributes=None, procedures=None, mod_name=None, type_name=None):
        Fortran.__init__(self, name, filename, doc, lineno)
        self.type = type
        self.attributes = attributes if attributes else []
        self.procedures = procedures if procedures else []
        self.mod_name = mod_name
        self.type_name = type_name


def iter_fields(node):
    """
    Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
    that is present on *node*.
    """
    for field in node._fields:
        try:
            yield (field, getattr(node, field))
        except AttributeError:
            pass

def iter_child_nodes(node):
    """
    Yield all direct child nodes of *node*, that is, all fields that are nodes
    and all items of fields that are lists of nodes.
    """
    for name, field in iter_fields(node):
        if isinstance(field, Fortran):
            yield field
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, Fortran):
                    yield item

def walk(node):
    """
    Recursively yield all descendant nodes in the tree starting at *node*
    (including *node* itself), in no specified order.  This is useful if you
    only want to modify nodes in place and don't care about the context.
    """
    from collections import deque
    todo = deque([node])
    while todo:
        node = todo.popleft()
        todo.extend(iter_child_nodes(node))
        yield node

def walk_modules(node):
    """
    Recursively yield all modules in the tree starting at *node*.
    """
    for child in walk(node):
        if isinstance(child, Module):
            yield child

def find_procedure_module(tree, node):
    """
    Find the module in `tree` that contains `node`
    """
    for mod in walk_modules(tree):
        if node in mod.procedures:
            return mod
        for intf in mod.interfaces:
            if node in intf.procedures:
                return mod
        for typ in mod.types:
            if node in typ.procedures:
                return mod
            for intf in typ.interfaces:
                if node in intf.procedures:
                    return mod
    return None


def walk_procedures(tree, include_ret_val=True):
    """
    Walk over all nodes in tree and yield tuples
    (module, procedure, arguments).

    If `include_ret_val` is true then Function return values are
    inserted after last non-optional argument.
    """
    for node in walk(tree):
        if not isinstance(node, Procedure):
            continue

        arguments = node.arguments[:]
        if include_ret_val and isinstance(node, Function):
            arguments.append(node.ret_val)

        mod = find_procedure_module(tree, node)

        yield (mod, node, arguments)

def find(tree, pattern):
    """
    Find a node whose name includes *pattern*
    """
    for node in walk(tree):
        if pattern.search(node.name):
            yield node


class FortranVisitor(object):
    """
    Implementation of the Visitor pattern for a Fortran parse tree.

    Walks the tree calling a visitor function for every node found. The
    visitor methods should be defined in subclasses as ``visit_`` plus the
    class name of the node, e.g. ``visit_Module`. If no visitor function is
    found the `generic_visit` visitor is used instead.
    """

    def visit(self, node):
        candidate_methods = ['visit_' + cls.__name__ for cls in
                             node.__class__.__mro__]
        for method in candidate_methods:
            try:
                visitor = getattr(self, method)
                break
            except AttributeError:
                continue
        else:
            visitor = self.generic_visit

        result = visitor(node)
        return result

    def generic_visit(self, node):
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, Fortran):
                        self.visit(item)
            elif isinstance(value, Fortran):
                self.visit(value)


class FortranTransformer(FortranVisitor):
    """
    Subclass of `FortranVisitor` which allows tree to be modified.

    Walks the Fortran parse tree and uses the return value of the
    visitor methods to replace or remove old nodes. If the return
    value of the visitor method is ``None``, the node will be removed.
    """

    def generic_visit(self, node):
        for field, old_value in iter_fields(node):
            old_value = getattr(node, field, None)
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, Fortran):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, Fortran):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, Fortran):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class FortranTreeDumper(FortranVisitor):
    """
    Subclass of `FortranVisitor` which prints a textual representation
    of the Fortran parse tree.
    """

    def __init__(self):
        self.depth = 0
    def generic_visit(self, node):
        print('  ' * self.depth + str(node))
        self.depth += 1
        FortranVisitor.generic_visit(self, node)
        self.depth -= 1

def dump(node):
    """Print contents of Fortran parse tree starting at `node`."""
    FortranTreeDumper().visit(node)

def find_source(node):
    """Locate source code for *node*. Returns a list of strings."""
    if node.filename == '' or node.filename is None:
        return
    lines = open(node.filename, 'r').readlines()
    if isinstance(node.lineno, slice):
        lineno = slice(node.lineno.start - 1, node.lineno.stop - 1)
    else:
        lineno = node.lineno - 1
    return lines[lineno]

def print_source(node, out=None):
    """Print source code for node to *out* (default is sys.stdout)."""
    if out is None:
        import sys
        out = sys.stdout
    source = find_source(node)
    out.writelines(source)

def find_types(tree, skipped_types=None):
    """
    Walk over all the nodes in tree, building up a dictionary:
      types: maps type names to Type instances

    Returns a pair (types, types_to_mod_names)
    """
    types = {}

    if skipped_types is None:
        skipped_types = []

    for mod in walk_modules(tree):
        for node in walk(mod):
            if isinstance(node, Type):
                if node.name not in skipped_types:
                    log.debug('type %s defined in module %s' % (node.name, mod.name))
                    node.mod_name = mod.name  # save module name in Type instance
                    node.uses = set([(mod.name, (node.name,))])
                    types[node.name] = node
                    types['type(%s)' % node.name] = node
                    types['class(%s)' % node.name] = node
                else:
                    log.info('Skipping type %s defined in module %s' % (node.name, mod.name))

    return types

def fix_argument_attributes(node):
    """
    Walk over all procedures in the tree starting at `node` and
    fix the argument attributes.
    """
    for mod, sub, arguments in walk_procedures(node):
        for arg in arguments:
            if not hasattr(arg, 'type'):
                arg.type = 'callback'
                arg.value = ''
                arg.attributes.append('callback')

    return node


class LowerCaseConverter(FortranTransformer):
    """
    Subclass of FortranTransformer which converts program, module,
    procedure, interface, type and declaration names and attributes to
    lower case. Original names are preserved in the *orig_name*
    attribute.
    """

    def visit_Program(self, node):
        node.orig_name = node.name
        node.name = node.name.lower()
        return self.generic_visit(node)

    def visit_Module(self, node):
        node.orig_name = node.name
        node.name = node.name.lower()
        node.default_access = node.default_access.lower()
        node.private_symbols = [p.lower() for p in node.private_symbols]
        node.public_symbols = [p.lower() for p in node.public_symbols ]
        node.uses = [u.lower() for u in node.uses]
        return self.generic_visit(node)

    def visit_Procedure(self, node):
        node.orig_name = node.name
        node.name = node.name.lower()
        node.uses = [u.lower() for u in node.uses]
        if node.mod_name is not None:
            node.mod_name = node.mod_name.lower()
        return self.generic_visit(node)

    def visit_Interface(self, node):
        node.orig_name = node.name
        node.name = node.name.lower()
        if node.mod_name is not None:
            node.mod_name = node.mod_name.lower()
        return self.generic_visit(node)

    def visit_Binding(self, node):
        if node.type_name is not None:
            node.type_name = node.type_name.lower()
        if node.mod_name is not None:
            node.mod_name = node.mod_name.lower()
        return self.generic_visit(node)

    def visit_Type(self, node):
        node.orig_name = node.name
        node.name = node.name.lower()
        node.attributes = [a.lower() for a in node.attributes]
        if node.mod_name is not None:
            node.mod_name = node.mod_name.lower()
        return self.generic_visit(node)

    def visit_Declaration(self, node):
        node.orig_name = node.name
        node.name = node.name.lower()
        node.type = node.type.lower()
        node.attributes = [a.lower() for a in node.attributes]
        return self.generic_visit(node)

    def visit_Element(self, node):
        node.orig_name = node.name
        node.name = node.name.lower()
        node.type = node.type.lower()
        node.attributes = [a.lower() for a in node.attributes]
        return self.generic_visit(node)


class RepeatedInterfaceCollapser(FortranTransformer):
    """
    Collapse repeated interfaces with the same name into a single interface
    """

    def visit_Module(self, node):
        interface_map = {}
        for interface in node.interfaces:
            if interface.name in interface_map:
                interface_map[interface.name].procedures.extend(interface.procedures)
            else:
                interface_map[interface.name] = interface
        node.interfaces = list(interface_map.values())
        return self.generic_visit(node)

def strip_type(t):
    """Return type name from type declaration"""
    t = t.replace(' ', '')  # remove blanks
    if t.startswith('type('):
        t = t[t.index('(') + 1:t.index(')')]
    if t.startswith('class('):
        t = t[t.index('(') + 1:t.index(')')]
    return t.lower()

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
                    log.debug('marking public symbol ' + node.name)
                    self.mod.public_symbols.append(node.name)
            else:
                # symbol should be marked as private if it's not already
                if node.name not in self.mod.private_symbols:
                    log.debug('marking private symbol ' + node.name)
                    self.mod.private_symbols.append(node.name)

        elif self.mod.default_access == 'private':
            if ('public' not in getattr(node, 'attributes', {}) and
                   node.name not in self.mod.public_symbols):

                # symbol should be marked as private if it's not already
                if node.name not in self.mod.private_symbols:
                    log.debug('marking private symbol ' + node.name)
                    self.mod.private_symbols.append(node.name)
            else:
                # symbol should be marked as public if it's not already
                if node.name not in self.mod.public_symbols:
                    log.debug('marking public symbol ' + node.name)
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
            log.debug('removing private symbol ' + node.name)
            return None
        else:
            return node

def remove_private_symbols(node):
    """
    Walk the tree starting at *node*, removing all private symbols.

    This function first applies the AccessUpdater transformer to
    ensure module *public_symbols* and *private_symbols* are up to
    date with *default_access* and individual `public` and `private`
    attributes.
    """

    node = AccessUpdater().visit(node)
    node = PrivateSymbolsRemover().visit(node)
    return node

type_re = re.compile(r'(type|class)\s*\((.*?)\)')
def derived_typename(typename):
    """
    type(TYPE) -> TYPE
    class(TYPE) -> TYPE
    otherwise -> None
    """
    m = type_re.match(typename)
    return m.group(2) if m else None

def is_derived_type(typename):
    return type_re.match(typename) != None

def split_type_kind(typename):
    """
    type*kind -> (type, kind)
    type(kind) -> (type, kind)
    type(kind=kind) -> (type, kind)
    """
    if '*' in typename:
        type = typename[:typename.index('*')]
        kind = typename[typename.index('*') + 1:]
    elif '(' in typename:
        type = typename[:typename.index('(')]
        kind = typename[typename.index('('):]
    else:
        type = typename
        kind = ''
    kind = kind.replace('kind=', '')
    return (type.strip(), kind.strip())


def f2c_type(typename, kind_map):
    """
    Convert string repr of Fortran type to equivalent C type

    Kind constants defined in `kind_map` are expanded, and a RuntimeError
    is raised if a undefined (type, kind) combination is encountered.
    """

    # default conversion from fortran to C types
    default_f2c_type = {
        'character': 'char',
        'integer': 'int',
        'real': 'float',
        'double precision': 'double',
        'logical': 'int',
        }

    type, kind = split_type_kind(typename)
    kind = kind.replace('(', '').replace(')', '')


    if type in kind_map:
        if kind in kind_map[type]:
            c_type = kind_map[type][kind]
        else:
            raise RuntimeError('Unknown combination of type "%s" and kind "%s"' % (type, kind) +
                               ' - add to kind map and try again')
    else:
        if type in default_f2c_type:
            c_type = default_f2c_type[type]
        elif type.startswith('type'):
            return 'type'
        elif type.startswith('class'):
            return 'type'
        else:
            raise RuntimeError('Unknown type "%s" - ' % type +
                               'add to kind map and try again')
    return c_type


def normalise_type(typename, kind_map):
    """
    Normalise Fortran type names, expanding kind constants defined in kind_map

    real(kind=dp) -> real(8), etc.
    """
    type, kind = split_type_kind(typename)
    if not kind:
        return type
    c_type = f2c_type(typename, kind_map)
    c_type_to_fortran_kind = {
        'char' : '',
        'signed_char' : '',
        'short' : '(2)',
        'int' : '(4)',
        'long_long' : '(8)',
        'float' :  '(4)',
        'double' : '(8)',
        'long_double' : '(16)',
        'complex_float' : '(4)',
        'complex_double' : '(8)',
        'complex_long_double' : '(16)',
        'string' : '',
        }
    orig_kind = kind
    kind = c_type_to_fortran_kind.get(c_type, kind)
    # special case: preserve string lengths
    if c_type == 'char':
        kind = orig_kind
    return type + kind


def fortran_array_type(typename, kind_map):
    """
    Convert string repr of Fortran type to equivalent numpy array typenum
    """
    c_type = f2c_type(typename, kind_map)

    # convert from C type names to numpy dtype strings
    c_type_to_numpy_type = {
        'char' : 'uint8',
        'signed_char' : 'int8',
        'short' : 'int16',
        'int' : 'int32',
        'long_long' : 'int64',
        'float' :  'float32',
        'double' : 'float64',
        'long_double' : 'float128',
        'complex_float' : 'complex64',
        'complex_double' : 'complex128',
        'complex_long_double' : 'complex256',
        'string' : 'str',
    }

    if c_type not in c_type_to_numpy_type:
        raise RuntimeError('Unknown C type %s' % c_type)

    # find numpy numerical type code
    numpy_type = np.dtype(c_type_to_numpy_type[c_type]).num
    return numpy_type

def f2py_type(type, attributes=None):
    """
    Convert string repr of Fortran type to equivalent Python type
    """
    if attributes is None:
        attributes = []
    if "real" in type:
        pytype = "float"
    elif "integer" in type:
        pytype = "int"
    elif "character" in type:
        pytype = 'str'
    elif "logical" in type:
        pytype = "bool"
    elif "complex" in type:
        pytype = 'complex'
    elif type.startswith("type"):
        pytype = strip_type(type).title()
    else:
        pytype = "unknown"
    dims = list(filter(lambda x: x.startswith("dimension"),
                  attributes))
    if len(dims) > 0:
        pytype += " array"
    return pytype
