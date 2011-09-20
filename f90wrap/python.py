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

from fortran import (Fortran, Program, Module, Subroutine, Function,
                     Declaration, Type, Interface, FortranVisitor)

class PythonCodeGenerator(CodeVisitor, CodeGenerator):

    def __init__(self, callbacks, kinds, types):
        self.callbacks = callbacks
        self.kinds = kinds
        self.types = types
        CodeGenerator.__init__(self,
                               indent='    ',
                               continuation='\\',
                               max_length=80)


    def visit_Subroutine(self, node):
        if not self.is_wrappable(mod, sub):
            return
        
        self.write("def %(sub_name)s(%(arg_names)s):" %
                 {'sub_name': sub.name,
                  'arg_names': arg_names(sub)})
        self.indent()
        self.write('pass')
        self.dedent()
        self.blank()
