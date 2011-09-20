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

import weakref
import logging

class Fortran(object):
    """
    Abstract base class for all nodes in Fortran parser tree. Has
    attributes *parent* (weak reference to parent node or None),
    *name*, *filename*, *doc*, *lineno*.
    """

    _fields = []

    def __init__(self, parent=None, name='', filename='', doc=None,
                 lineno=0):
        self.parent = parent
        self.name = name
        self.filename = filename
        if doc is None:
            doc = []
        self.doc = doc
        self.lineno = lineno

    def __repr__(self):
        return '%s(name=%s)' % (self.__class__.__name__, self.name)

    def __eq__(self, other):
        if other is None: return False
        return (self.name == other.name and
                self.doc == other.doc)

    def __neq__(self, other):
        return not self.__eq__(self, other)


class Root(Fortran):
    """
    Root node of a Fortran parse tree. Has attributes *programs*, *modules*
    and *procedures* in addition to those in Fortran base class.
    """

    _fields = ['programs', 'modules', 'procedures']

    def __init__(self, parent=None, name='', filename='', doc=None, lineno=0,
                 programs=None, modules=None, procedures=None):
        Fortran.__init__(self, parent, name, filename, doc, lineno)
        if programs is None:
            programs = []
        self.programs = programs
        if modules is None:
            modules = []
        self.modules = modules
        if procedures is None:
            procedures = []
        self.procedures = procedures

    def __eq__(self, other):
        if other is None: return False
        return (self.programs == other.programs and
                self.modules == other.modules and
                self.doc == other.doc and
                self.procedures == other.procedures)

    def __neq__(self, other):
        return not self.__eq__(other)


class Program(Fortran):
    """
    Class to represent a Fortran main program. Has *procedures* attribute
    in addition to Fortran base class attributes.
    """

    _fields = ['procedures']

    def __init__(self, parent=None, name='', filename='', doc=None, lineno=0,
                 procedures=None):
        Fortran.__init__(self, parent, name, filename, doc, lineno)
        if procedures is None:
            procedures = []
        self.procedures = procedures

    def __eq__(self, other):
        if other is None: return False
        return (self.name == other.name and
                self.doc == other.doc and
                self.procedures == other.procedures and
                self.uses == other.uses)

    def __ne__(self, other):
        return not self.__eq__(other)


class Module(Fortran):
    """
    Represents a Fortran module. Attributes in addition those of Fortran
    base class are *types*, *elements*, *procedures*, *interfaces*,
    *uses*, *default_access*, *public_symbols* and *private_symbols*.
    """

    _fields = ['types', 'elements', 'procedures', 'interfaces', 'uses']

    def __init__(self, parent=None, name='', filename='', doc=None, lineno=0,
                 types=None, elements=None, procedures=None,
                 interfaces=None, uses=None, default_access='public',
                 public_symbols=None, private_symbols=None):
        Fortran.__init__(self, parent, name, filename, doc, lineno)
        if types is None:
            types = []
        self.types = types
        if elements is None:
            elements = []
        self.elements = elements
        if procedures is None:
            procedures = []
        self.procedures=procedures
        if interfaces is None:
            interfaces = []
        self.interfaces=interfaces
        if uses is None:
            uses = []
        self.uses=uses
        self.default_access = default_access
        if public_symbols is None:
            public_symbols = []
        self.public_symbols = public_symbols
        if private_symbols is None:
            private_symbols = []
        self.private_symbols = private_symbols

    def __eq__(self, other):
        if other is None: return False
        return (self.name == other.name and
                self.types == other.types and
                self.elements == other.elements and
                self.procedures == other.procedures and
                self.doc == other.doc and
                self.uses == other.uses and
                self.interfaces == other.interfaces and
                self.default_access == other.default_access and
                self.public_symbols == other.public_symbols and
                self.private_symbols == other.private_symbols)

    def __ne__(self, other):
        return not self.__eq__(other)

class Procedure(Fortran):
    """
    Abstract class for representing subroutines and functions.
    *arguments* attribute is list of routine arguments.
    """

    _fields = ['arguments']

    def __init__(self, parent=None, name='', filename='', doc=None, lineno=0,
                 arguments=None, uses=None, recur=''):
        Fortran.__init__(self, parent, name, filename, doc, lineno)
        if arguments is None: arguments = []
        self.arguments = arguments
        if uses is None: uses = []
        self.uses = uses
        self.recur = recur

    def __eq__(self, other):
        if other is None: return False
        return (self.name == other.name and
                self.arguments == other.arguments and
                self.doc == other.doc and
                self.uses == other.uses and
                self.recur == other.recur)

    def __ne__(self, other):
        return not self.__eq__(other)

class Subroutine(Procedure):
    """
    Subclass of Procedure to represent a Fortran subroutine.
    """

    pass

class Function(Procedure):
    """
    Subclass of Procedure to represent a Fortran function.
    Additional attributes *ret_val* and *ret_val_doc*.
    """

    _fields = ['arguments', 'ret_val']

    def __init__(self, parent=None, name='', filename='', doc=None, lineno=0,
                 arguments=None, uses=None, recur='',
                 ret_val=None, ret_val_doc=None):
        Procedure.__init__(self, parent, name, filename, doc,
                           lineno, arguments, uses, recur)
        if ret_val is None:
            ret_val = Argument()
        self.ret_val = ret_val
        self.ret_val_doc = ret_val_doc

    def __eq__(self, other):
        if other is None: return False
        return (self.name == other.name and
                self.arguments == other.arguments and
                self.doc == other.doc and
                self.uses == other.uses and
                self.ret_val == other.ret_val and
                self.ret_val_doc == other.ret_val_doc and
                self.recur == other.recur)

    def __ne__(self, other):
        return not self.__eq__(other)


class Prototype(Fortran):
    """
    Procedure prototype. Used to populate Interfaces before subroutines
    and functions are added to them.
    """

    pass

class Declaration(Fortran):
    """
    Variable declaration. Parent is either a Module, a Type
    or a Procedure. Additional attributes *type*,
    *attributes* and *value*.
    """

    def __init__(self, parent=None, name='', filename='', doc=None, lineno=0,
                 attributes=None, type='', value=''):
        Fortran.__init__(self, parent, name, filename, doc, lineno)
        if attributes is None: attributes = []
        self.attributes = attributes
        self.type = type
        self.value = value

    def __eq__(self, other):
        if other is None: return False
        return (self.name == other.name and
                self.type == other.type and
                self.attributes == other.attributes and
                self.doc == other.doc and
                self.value == other.value)

    def __ne__(self, other):
        return not self.__eq__(other)

class Element(Declaration):
    pass

class Argument(Declaration):
    pass

class Type(Fortran):
    """
    Representation of a Fortran derived type. Additional attributes
    *elements* and *methods*.
    """

    _fields = ['elements', 'methods']

    def __init__(self, parent=None, name='', filename='', doc=None,
                 lineno=0, elements=None, methods=None):
        Fortran.__init__(self, parent, name, filename, doc, lineno)
        if elements is None: elements = []
        self.elements = elements
        if methods is None: methods = []
        self.methods = methods

    def __eq__(self, other):
        if other is None: return False
        return (self.name == other.name and
                self.elements == other.elements and
                self.doc == other.doc)

    def __ne__(self, other):
        return not self.__eq__(other)


class Interface(Fortran):
    """
    Represenation of a Fortran interface. Additional attribute
    *procedures*.
    """

    _fields = ['procedures']

    def __init__(self, parent=None, name='', filename='', doc=None,
                 lineno=0, procedures=None):
        Fortran.__init__(self, parent, name, filename, doc, lineno)
        if procedures is None: procedures = []
        self.procedures = procedures

    def __eq__(self, other):
        if other is None: return False
        return (self.name == other.name and
                self.procedures == other.procedures and
                self.doc == other.doc)

    def __ne__(self, other):
        return not self.__eq__(other)



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

def find_parent_module(node):
    """
    Return the Module in which *node* appears, or None if it's defined
    at top-level Goes up the tree starting at *node* until either a
    module parent or the root is located.
    """

    while True:
        if node.parent is None:
            return None
        node = node.parent()
        if isinstance(node, Module) or node is None:
            return node

def walk_procedures(tree, include_ret_val=True, skip_if_outside_module=True):
    """
    Walk over all nodes in tree and yield tuples
    (module, procedure, arguments).

    If `include_ret_val` is true then Function return values are
    inserted after last non-optional argument. If
    `skip_if_outside_module` is True, top-level subroutines and
    functions are not included.
    """
    for node in walk(tree):
        if isinstance(node, Procedure):
            mod = find_parent_module(node)
            if mod is None and skip_if_outside_module:
                continue

            arguments = node.arguments[:]
            if include_ret_val and isinstance(node, Function):
                arguments.append(node.ret_val)

            yield (mod, node, arguments)


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
        print '  '*self.depth + str(node)
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
        lineno = slice(node.lineno.start-1,node.lineno.stop-1)
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

def fix_parents(node):
    """Update parent references in the tree starting at *node*."""
    def _fix(node):
        for child in iter_child_nodes(node):
            child.parent = weakref.ref(node)
            _fix(child)
    _fix(node)
    return node

def find_types(tree):
    """
    Walk over all the nodes in tree, building a map from
    type to module names.
    """
    types = {}
    for node in walk(tree):
        if isinstance(node, Type):
            mod = find_parent_module(node)
            logging.debug('type %s defined in module %s' %
                          (node.name, mod.name))
            types['type(%s)' % node.name] = types[node.name] = mod.name
    return types


def fix_argument_attributes(node):
    """
    Walk over all procedures in the tree starting at `node` and
    fix the argument attributes.
    """
    for mod, sub, arguments in walk_procedures(node):
        for arg in arguments:
            if not hasattr(arg, 'attributes'):
                arg.is_callback = True
                arg.attributes = []
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
        node.public_symbols  = [p.lower() for p in node.public_symbols ]
        node.uses = [u.lower() for u in node.uses]
        return self.generic_visit(node)

    def visit_Subroutine(self, node):
        node.orig_name = node.name
        node.name = node.name.lower()
        node.uses = [u.lower() for u in node.uses]
        return self.generic_visit(node)

    visit_Function = visit_Subroutine

    def visit_Interface(self, node):
        node.orig_name = node.name
        node.name = node.name.lower()
        return self.generic_visit(node)

    def visit_Type(self, node):
        node.orig_name = node.name
        node.name = node.name.lower()
        return self.generic_visit(node)

    def visit_Declaration(self, node):
        node.orig_name = node.name
        node.name = node.name.lower()
        node.type = node.type.lower()
        node.attributes = [a.lower() for a in node.attributes]
        return self.generic_visit(node)


class AccessUpdater(FortranTransformer):
    """Visit module contents and update public_symbols and
       private_symbols lists to be consistent with (i) default module
       access; (ii) public and private statements at module level;
       (iii) public and private attibutes."""

    def visit(self, node):
        mod = find_parent_module(node)
        if mod is None:
            return self.generic_visit(node)

        if mod.default_access == 'public':
            if ('private' not in getattr(node, 'attributes', {}) and
                   node.name not in mod.private_symbols):

                # symbol should be marked as public if it's not already
                if node.name not in mod.public_symbols:
                    logging.debug('marking public symbol '+ node.name)
                    mod.public_symbols.append(node.name)
            else:
                # symbol should be marked as private if it's not already
                if node.name not in mod.private_symbols:
                    logging.debug('marking private symbol '+ node.name)
                    mod.private_symbols.append(node.name)

        elif mod.default_access == 'private':
            if ('public' not in getattr(node, 'attributes', {}) and
                   node.name not in mod.public_symbols):

                # symbol should be marked as private if it's not already
                if node.name not in mod.private_symbols:
                    logging.debug('marking private symbol '+ node.name)
                    mod.private_symbols.append(node.name)
            else:
                # symbol should be marked as public if it's not already
                if node.name not in mod.public_symbols:
                    logging.debug('marking public symbol '+ node.name)
                    mod.public_symbols.append(node.name)

        else:
            raise ValueError('bad default access %s for module %s' %
                               (mod.default_access, mod.name))

        return node # no need to recurse further


class PrivateSymbolsRemover(FortranTransformer):
    """
    Transform a tree by removing private symbols.
    """

    def visit(self, node):
        mod = find_parent_module(node)
        if mod is None:
            return self.generic_visit(node)

        if node.name in mod.private_symbols:
            logging.debug('removing private symbol '+node.name)
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
