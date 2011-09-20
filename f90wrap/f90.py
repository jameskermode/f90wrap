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
from parser import read_files
from fortran import *
from codegen import CodeGenerator

class F90WrapperGenerator(FortranVisitor, CodeGenerator):

    def __init__(self, prefix, kinds, types):
        CodeGenerator.__init__(self, indent='    ',
                               max_length=80,
                               continuation='&')
        FortranVisitor.__init__(self)
        self.prefix = prefix
        self.kinds = kinds
        self.types = types

    def visit_Module(self, node):
        self.code = []
        self.generic_visit(node)
        f90_wrapper_file = open('%s%s.f95' % (self.prefix, node.name), 'w')
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
            arg_dict = {'arg_type': arg.type,
                        'comma': len(arg.attributes) != 0 and ', ' or '',
                        'arg_attribs': ', '.join(arg.attributes),
                        'arg_name': self.prefix+arg.name}
            if arg.name in node.transfer_in or arg.name in node.transfer_out:
                self.write('type(%(arg_type)s)_ptr :: %(arg_name)s_ptr' % arg_dict)
                arg_dict['arg_type'] = arg.wrapper_type
            self.write('%(arg_type)s%(comma)s%(arg_attribs)s :: %(arg_name)s' % arg_dict)                            
        self.write('! END write_arg_decl_lines ')
        self.write()        

    def write_transfer_in_lines(self, node):
        self.write('! BEGIN write_transfer_in_lines ')        
        for arg in node.arguments:
            arg_dict = {'arg_name': self.prefix+arg.name,
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
            self.write('allocate(%s)' % alloc)
        for arg in node.arguments:
            if not hasattr(arg, 'init_lines'):
                continue
            exe_optional, exe = arg.init_lines                
            D = {'OLD_ARG':arg.name,
                 'ARG':self.prefix+arg.name,
                 'PTR':arg.name+'_ptr%p'}
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

        arg_names = ['%s=%s%s' % (arg.name,self.prefix,arg.name) for arg in node.arguments]
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
            self.write('deallocate(%s)' % dealloc)
        self.write('! END write_finalise_lines')
        self.write()

    def visit_Subroutine(self, node):

        self.write("subroutine %(sub_name)s(%(arg_names)s)" %
                   {'sub_name': self.prefix+node.name,
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
        self.write("end subroutine %(sub_name)s" % {'sub_name': self.prefix+node.name})
        self.write()
        self.write()


class UnwrappablesRemover(FortranTransformer):

    def __init__(self, callbacks, types):
        self.callbacks = callbacks
        self.types = types

    def visit_Subroutine(self, node):
        args = node.arguments[:]
        if isinstance(node, Function):
            args.append(node.ret_val)

        for arg in args:
            # only callback functions in self.callbacks
            if not hasattr(arg, 'type') or not hasattr(arg, 'attributes'):
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

    visit_Function = visit_Subroutine

    def visit_Declaration(self, node):
        if isinstance(node.parent(), Subroutine):
            # node is a subroutine or function argument
            
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



class MethodFinder(FortranTransformer):

    def __init__(self):
        FortranTransformer.__init__(self)

    def visit_Type(self, node):
        pass

def fix_subroutine_uses_clauses(tree, types, kinds):
    """Walk over all nodes in tree, updating subroutine uses
       clauses to include the parent module and all necessary modules
       from types and kinds."""
       
    for mod,sub,arguments in walk_procedures(tree):

        sub.uses = set()
        sub.uses.add((mod.name, None))

        for arg in arguments:
            if arg.type.startswith('type') and arg.type in types:
                sub.uses.add((types[arg.type], None))

        for (mod, only) in kinds:
            if mod not in sub.uses:
                sub.uses.add((mod, only))

def wrap_derived_types(tree, init_lines):
    for mod,sub,arguments in walk_procedures(tree):
        sub.types = set()
        sub.transfer_in = []
        sub.transfer_out = []
        sub.finalise_lines = []
        sub.allocate = []
        sub.deallocate = []
        for arg in arguments:
            if not arg.type.startswith('type'):
                continue
            
            typename = arg.type[5:-1]
            arg.wrapper_type = 'integer(SIZEOF_FORTRAN_T)'
            sub.types.add(typename)

            if typename in init_lines:
                 use, (exe, exe_optional) = init_lines[typename]
                 if use is not None:
                     sub.uses.add((use, None))
                 arg.init_lines = (exe_optional, exe)

            if 'intent(in)' in arg.attributes or 'intent(inout)' in arg.attributes:
                sub.transfer_in.append(arg.name)
            if 'intent(out)' in arg.attributes:
                sub.transfer_out.append(arg.name)
                sub.allocate.append(arg.name)


class FunctionToSubroutineConverter(FortranTransformer):
    """Convert all functions to subroutines, with return value
       as intent(out) argument after last non-optional argument"""

    def visit_Function(self, node):

        # insert ret_val after last non-optional argument
        arguments = node.arguments[:]
        for i, arg in enumerate(arguments):
            if 'optional' in arg.attributes:
                break
        arguments.insert(i, node.ret_val)
        arguments[i].name = 'ret_'+arguments[i].name
        arguments[i].attributes.append('intent(out)')
        
        new_node = Subroutine(node.parent,
                              node.name,
                              node.filename,
                              node.doc,
                              arguments,
                              node.uses,
                              node.recur)
        new_node.orig_node = node # keep a reference to the original node
        return new_node



