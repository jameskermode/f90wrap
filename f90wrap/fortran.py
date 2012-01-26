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
from fparser.api import *
from fparser.base_classes import Statement, Variable
from fparser.block_statements import BeginStatement, SubProgramStatement # common ancestor of Subroutine and Function, not in __all__

fortran_base_classes = (Statement, Variable)
variable_attributes = ['name', 'typedecl', 'dimension', 'bounds', 'length',
                       'attributes', 'bind', 'intent', 'check', 'init']

def walk_class_instances(node, cls):
    for child, depth in walk(node):
        if isinstance(child, cls):
            yield child

def walk_modules(node):
    """
    Recursively yield all modules in the tree starting at *node*.
    """
    for mod in walk_class_instances(node, Module):
        yield mod

def walk_procedures(tree, include_ret_val=True):
    """
    Walk over all nodes in tree and yield tuples
    (module, procedure, arguments).

    If `include_ret_val` is True (default) then Function return values are
    inserted at beginning of argument list.
    """
    for node in walk_class_instances(tree, SubProgramStatement):
        arguments = node.a.variables.values()
        if isinstance(node, Function) and not include_ret_val:
            arguments = arguments[1:]
        yield (node.parent, node, arguments)

def find(tree, pattern):
    for node, depth in walk(tree):
        try:
            if pattern.search(node.name):
                yield node
        except AttributeError:
            continue


class Visitor(object):
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
        for child in node.content:
            self.visit(child)

class Transformer(Visitor):
    """
    Subclass of `Visitor` which allows tree to be modified.

    Walks the parse tree and uses the return value of the visitor
    methods to replace or remove old nodes. If the return value of the
    visitor method is ``None``, the node will be removed.
    """

    def generic_visit(self, node):
        new_content = []
        for child in node.content:
            child = self.visit(child)
            if child is None:
                continue
            elif not isinstance(child, fortran_base_classes):
                new_content.extend(child)
                continue
            new_content.append(child)
        node.content[:] = new_content
        #node.analyze() ?
        return node

def dump(node):
    """Print contents of Fortran parse tree starting at `node`."""
    for stmt, depth in walk(node):
        print str(stmt)

def find_source(node):
    """Locate source code for *node*."""
    while not hasattr(node, 'item'):
        node = node.parent
    
    if isinstance(node, BeginStatement):
        span = (node.item.span[0], node.content[-1].item.span[1]+1)
    else:
        span = (node.item.span[0], node.item.span[1]+1)
    s = slice(span[0]-1, span[1]-1, None)
    return node.item.reader.file.name, span, node.item.reader.source_lines[s]
    
def print_source(node, out=None, linenumbers=True):
    """Print source code for node to *out* (default is sys.stdout)."""
    if out is None:
        import sys
        out = sys.stdout
    file, span, source_lines = find_source(node)
    if linenumbers:
        for lineno, source_line in zip(range(*span), source_lines):
            out.write('%6d%s\n' % (lineno, source_line))
    else:
        out.write('\n'.join(source_lines))
                

def find_types(tree):
    """
    Walk over all the nodes in tree, building up a dictionary:
    mapping type names to Type instances
    """
    types = {}

    for node in walk_class_instances(tree, TypeDecl):
        mod = node.parent
        node.info('type %s defined in module %s' % (node.name, mod.name))
        node.mod_name = mod.name  # save module name in Type instance
        types['type(%s)' % node.name] = types[node.name] = node
            
    return types


