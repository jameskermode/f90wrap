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

class CodeGenerator(object):
    """
    Simple class to handle code generation.
    
    Handles simple tasks such as indent/dedent and continuation symbols.
    
    Parameters
    ----------
    indent : `str`
        Specification of the indent size/type. Typical choices may be ``" "*4``
        or ``"\t"``.
        
    max_length : `int`
        Maximum length of a code line.
        
    continuation : `str`
        Symbol to represent continuation in the desired language (eg. '&' in 
        Fortran)

    comment : `str`
        Character used to define comments (e.g. '!' in Fortran, '#' in Python)
    """

    def __init__(self, indent, max_length, continuation, comment):
        self._indent = indent
        self.max_length = max_length
        self.continuation = continuation
        self.comment = comment
        self.level = 0
        self.code = []

    def indent(self):
        """Indent code level"""
        self.level += 1

    def dedent(self):
        """Dedent code level"""
        self.level -= 1

    def write(self, *args):
        """
        Write arbitrary string arguments to the instance's code, split by
        newline characters and implied newline after last arg.
        """
        if args is ():
            args = ('\n',)
        args = ' '.join(args).rstrip() + '\n'
        lines = args.splitlines(True)
        self.code.extend([self._indent * self.level + line for line in lines])

    def writelines(self, items, insert=None, level=None):
        """
        Write the given code lines to the instance's code. 
        
        Parameters
        ----------
        items : list of strings
            A list of code lines to be appended to the instance's code.
            Newline characters with strings will automatically be propagated
            into the code.
        insert : integer or None
            If present, insert lines after index `insert` rather than appending
        level : integer or None
            If present, override the current indent level

        Returns
        -------
        index : index for next line to be added (equal to len(code) if insert=None)
        """
        if level is None:
            level = self.level

        lines = []
        for item in items:
            item_lines = item.splitlines(True)
            if not item_lines[-1].endswith('\n'):
                item_lines[-1] += '\n'
            lines.extend(item_lines)

        lines = [self._indent * level + line for line in lines]
        if insert is not None:
            self.code = self.code[:insert] + lines + self.code[insert:]
            return insert + len(lines)
        else:
            self.code.extend(lines)
            return len(self.code)

    def split_long_lines(self):
        """
        Split lines longer than `max_length` using `continuation`

        Ignores lines starting with comment marker
        """
        out = []
        for line in self.code:
            if len(line) > self.max_length and not line.strip().startswith(self.comment):
                indent = line[:len(line) - len(line.lstrip())]
                tokens = line.split()
                split_lines = [[]]
                while tokens:
                    token = tokens.pop(0)
                    current_line = ' '.join(split_lines[-1])
                    if len(current_line) == 0 or len(current_line) + len(token) < self.max_length:
                        split_lines[-1].append(token)
                    else:
                        split_lines[-1].append(self.continuation)
                        split_lines.append([self._indent + token])
                out.extend([indent + ' '.join(line) + '\n' for line in split_lines])
            else:
                out.append(line)
        return out

    def __str__(self):
        return "".join(self.split_long_lines())
