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

class CodeGenerator(object):
    """Simple class for code generation"""

    def __init__(self, indent, max_length, continuation):
        self._indent = indent
        self.max_length = max_length
        self.continuation = continuation
        self.level = 0
        self.code = []

    def indent(self):
        self.level += 1

    def dedent(self):
        self.level -= 1

    def write(self, *args):
        "behaves like print statement, with implied \n after last arg"
        if args is ():
            args = ('\n',)
        args = ' '.join(args).rstrip()+'\n'
        lines = args.splitlines(True)
        self.code.extend([self._indent*self.level+line for line in lines])

    def writelines(self, items):
        lines = []
        for item in items:
            lines.extend(item.splitlines(True))
        self.code.extend([self._indent*self.level+line for line in lines])

    def split_long_lines(self):
        out = []
        for line in self.code:
            if len(line) > self.max_length:
                indent = line[:len(line)-len(line.lstrip())]
                tokens = line.split()
                split_lines = [[]]
                while tokens:
                    token = tokens.pop(0)
                    current_line = ' '.join(split_lines[-1])
                    if len(current_line) + len(token) < self.max_length:
                        split_lines[-1].append(token)
                    else:
                        split_lines[-1].append(self.continuation)
                        split_lines.append([self._indent+token])
                out.extend([indent+' '.join(line)+'\n' for line in split_lines])
            else:
                out.append(line)
        return out

    def __str__(self):
        return "".join(self.split_long_lines())
