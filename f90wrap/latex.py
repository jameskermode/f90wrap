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

# Originally based on:
# f90doc - automatic documentation generator for Fortran 90
# Copyright (C) 2004 Ian Rutt
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with this program; if not, write to the Free
# Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
# MA 02111-1307 USA

import re
import string
import copy
import logging
import os

from f90wrap import fortran as ft

import sys
major, minor = sys.version_info[0:2]

log = logging.getLogger(__name__)

if (major, minor) < (2, 5):
    all = lambda seq: not False in seq
    any = lambda seq: True in seq


latex_ = re.compile(r'([_])')
latex_special_chars = re.compile(r'([%#])')

def escape_code_sample(matchobj):
    s = matchobj.group(0).replace('$', r'\$')
    s = s.replace('{', '\{')
    s = s.replace('}', '\}')
    return s

class LatexOutput(object):

    def __init__(self):
        self.verbatim = False
        self.displaymath = False
        self.sections = ['\section', '\subsection*', '\subparagraph']

    def print_line(self, str):

        if str == '':
            self.stream.write('\n')
            return

        # Lines starting with '>' are to be printed verbatim
        if self.verbatim:
            if str[0] == '>':
                self.stream.write(str[1:] + '\n')
                return
            else:
                self.verbatim = False
                # print_line(r'\end{verbatim}')
                # print_line(r'\end{sidebar}')
                self.stream.write(r'''\end{verbatim}
                %\end{boxedminipage}

                        ''')
        else:
            if str[0] == '>':
                self.stream.write(r'''

                %\begin{boxedminipage}{\textwidth}
                \begin{verbatim}''')

    #            print_line(r'\begin{sidebar}')
    #            print_line(r'\begin{verbatim}')


                self.verbatim = True
                self.stream.write(str[1:] + '\n')
                return
            else:
                pass

        if self.displaymath:
            if re.search(r'\\end{(displaymath|equation|eqnarray)}', str):
                self.displaymath = False
        else:
            if re.search(r'\\begin{(displaymath|equation|eqnarray)}', str):
                self.displaymath = True

        # Escape latex special chars everywhere
        s = latex_special_chars.sub(r'\\\1', str)

        if not self.displaymath and not self.verbatim:
            # Escape '{' and '}' when between '...'
            # #L = re.split(r"\\'", s)
            # #L[::2] = [re.sub(r'([\{\}])',r'\\\1',p) for p in L[::2]]
            # #s = "'".join(L)

            # Put code examples in single quotes in \texttt{} font
            s = re.sub(r"\\'", r"\\quote\\", s)
            s = re.sub(r"'(.*?)'", escape_code_sample, s)

            s = re.sub(r"\\quote\\", r"'", s)

            # Escape '_' only when not between $...$
            L = re.split(r'\$', s)
            L[::2] = [latex_.sub(r'\\\1', p) for p in L[::2]]



            self.stream.write('$'.join(L) + '\n')

        else:
            self.stream.write(s + '\n')


def uniq(L, idfun=None):
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in L:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result


def combine_elements(elements):
    element_dict = {}
    func_args = []
    i = 0  # counter for appearance order of args
    for a in elements:
        if isinstance(a, ft.Subroutine) or isinstance(a, ft.Function):
            func_args.append(a)
            continue
        i = i + 1
        element_dict[a.name] = (a, i)

    # Combine names with the same type, attributes and doc string
    rev_dict = {}
    for type, name in zip([ x[0].type.lower() + str([y.lower for y in x[0].attributes]) + str(x[0].doc) \
                             for x in element_dict.values() ], element_dict.keys()):
        if rev_dict.has_key(type):
            rev_dict[type].append(element_dict[name])
        else:
            rev_dict[type] = [element_dict[name]]

    for k in rev_dict:
        names = [x[0].name for x in rev_dict[k]]
        a = rev_dict[k][0][0]
        names.sort(key=lambda x: element_dict[x][1])
        alist = []
        while names:
            n = 0
            length = 0
            while (length < 30 and n < len(names)):
                length = length + len(names[n])
                n = n + 1
            ns = names[:n]
            del names[:n]
            b = copy.copy(a)
            b.name = ', '.join(ns)
            alist.append(b)

        rev_dict[k] = (alist, min([x[1] for x in rev_dict[k]]))

    # Sort by original appearance order of first name
    keys = rev_dict.keys()
    keys.sort(key=lambda x: rev_dict[x][1])

    return keys, rev_dict, func_args


class LatexGenerator(ft.FortranVisitor, LatexOutput):

    def __init__(self, stream, fn='', short_doc=False):
        ft.FortranVisitor.__init__(self)
        LatexOutput.__init__(self)
        self.stream = stream
        self.depth = 0
        self.compact = False
        self.is_ret_val = False
        self.short_doc = short_doc


    def visit_Program(self, node):

        if node.doc:
            if node.doc[0].strip() == 'OMIT':
                return

            if node.doc[0].strip() == 'OMIT SHORT':
                if self.short_doc:
                    return
                else:
                    node.doc = node.doc[1:]


        self.print_line(r"\newpage")
        self.print_line(r'\index{general}{' + node.name + ' program}')
        self.print_line(self.sections[self.depth] + r'[Program \texttt{')
        self.print_line(node.name + '}]')
        self.print_line(r"""{Program \texttt{""")
        self.print_line(node.name)
        if self.depth == 0:
            self.print_line(r"""} in file """ + node.filename + """}""")
        else:
            self.print_line(r"}}")
        self.print_line(self.sections[self.depth + 1] + """{Purpose}""")
        for a in node.doc:
            # self.print_line( a)
            self.print_line(a)
        if node.uses != []:
            self.print_line(self.sections[self.depth + 1] + r"{Uses}")
            u_temp = ''
            for a in node.uses:
                u_temp = u_temp + a + ', '
                if len(u_temp) > 50:
                    self.print_line(r"\texttt{" + u_temp[:-2] + "}")
                    u_temp = ''
                    self.print_line("\n")
            if u_temp != '':
                self.print_line(r"\texttt{" + u_temp[:-2] + "}")
        self.depth += 1
        for a in node.procedures:
            self.visit(a)
        self.depth -= 1


    def visit_Module(self, node):

        if node.doc:
            if node.doc[0].strip() == 'OMIT':
                return

            if node.doc[0].strip() == 'OMIT SHORT':
                if self.short_doc:
                    return
                else:
                    node.doc = node.doc[1:]


        self.print_line(r"\newpage")
        self.print_line(r'\index{general}{' + node.name + ' module}')
        self.print_line(self.sections[self.depth] + r'[Module \texttt{')
        self.print_line(node.name + '}]')
        self.print_line(r"""{Module \texttt{""")
        self.print_line(node.name)
        if self.depth == 0:
            self.print_line(r"""} in file """ + node.filename + """}""")
        else:
            self.print_line(r"}}")
        self.print_line(self.sections[self.depth + 1] + """{Purpose}""")
        for a in node.doc:
            self.print_line(a)
        self.print_line(self.sections[self.depth + 1] + r'{Usage}')
        self.print_line('>    use ' + node.name)
        if node.uses != []:
            self.print_line(self.sections[self.depth + 1] + r"{Uses}")
            u_temp = ''
            for a in node.uses:
                u_temp = u_temp + a + ', '
                if len(u_temp) > 50:
                    self.print_line(r"\texttt{" + u_temp[:-2] + "}")
                    u_temp = ''
                    self.print_line("\n")
            if u_temp != '':
                self.print_line(r"\texttt{" + u_temp[:-2] + "}")

        if node.elements != []:
            self.print_line(self.sections[self.depth + 1] + r"""{Module variables}""")

            keys, rev_dict, func_args = combine_elements(node.elements)
            self.print_line(r"\begin{description}")
            for k in keys:
                for a in rev_dict[k][0]:
                    self.visit(a)
            self.print_line(r"\end{description}")

        self.depth += 1
        for a in node.types:
            self.visit(a)
        for a in node.interfaces:
            self.visit(a)
        for a in node.procedures:
            self.visit(a)
        self.depth -= 1


    def visit_Subroutine(self, node):

        if node.doc:
            if node.doc[0].strip() == 'OMIT':
                return

            if node.doc[0].strip() == 'OMIT SHORT':
                if short_doc:
                    return
                else:
                    node.doc = node.doc[1:]

        if self.compact:
            if node.arguments != []:
                argl = '(' + ','.join([x.name for x in node.arguments]) + ')'
            else:
                argl = ''

            d_ent = r"Subroutine \texttt{" + node.name + argl + "}"

            self.print_line(r"\item[" + d_ent + r"]\mbox{} \par\noindent")

            if node.arguments != []:
                self.print_line(r'\begin{description}')
                keys, rev_dict, func_args = combine_elements(node.arguments)
                for k in keys:
                    for a in rev_dict[k][0]:
                        self.visit(a)
                self.print_line(r'\end{description}')

            for a in node.doc:
                self.print_line(a)
            self.print_line('')

            return


        if node.arguments != []:
            argl = '('
            for a in range(len(node.arguments)):
                arg = node.arguments[a]
                if isinstance(arg, ft.Declaration) and 'optional' in arg.attributes:
                    if argl[-2:] == '],':
                        argl = argl[:-2] + ',' + arg.name.rstrip() + '],'
                    elif argl.rstrip()[-4:] == '], &':
                        argl = argl.rstrip()[:-4] + ', &\n                        ' + arg.name.rstrip() + '],'
                    elif argl[-1] == ',':
                        argl = argl[:-1] + '[,' + arg.name.rstrip() + '],'
                    else:
                        argl = argl + '[' + arg.name.rstrip() + '],'
                else:
                    argl = argl + arg.name.rstrip() + ','
                if (a + 1) % 4 == 0.0 and a + 1 != len(node.arguments):
                    argl = argl + ' &\n                        '
            argl = argl[:-1] + ')'
        else:
            argl = ''

        self.print_line(r'\index{general}{' + node.name + ' subroutine}')

        if 'recursive' not in node.attributes:
            self.print_line(self.sections[self.depth] + r""" {Subroutine \texttt{""" + node.name)
        else:
            self.print_line(self.sections[self.depth], r"""{Recursive subroutine \texttt{""", node.name)

        self.print_line("""}}""")
        self.print_line('>    call ' + node.name + argl)

        for a in node.doc:
            self.print_line(a)

        if node.uses != []:
            self.print_line(self.sections[self.depth + 1] + r"{Uses}")
            u_temp = ''
            for a in node.uses:
                u_temp = u_temp + a + ', '
                if len(u_temp) > 50:
                    self.print_line(r"\texttt{" + u_temp[:-2] + "}")
                    u_temp = ''
                    self.print_line("\n")
            if u_temp != '':
                self.print_line(r"\texttt{" + u_temp[:-2] + "}")

        if node.arguments != []:

            keys, rev_dict, func_args = combine_elements(node.arguments)

            self.print_line(r"\begin{description}")
            for k in keys:
                for a in rev_dict[k][0]:
                    self.visit(a)
            for f in func_args:
                self.compact = True
                self.visit(f)
                self.compact = False

            self.print_line(r"\end{description}")


    def visit_Function(self, node):

        if node.doc:
            if node.doc[0].strip() == 'OMIT':
                return

            if node.doc[0].strip() == 'OMIT SHORT':
                if short_doc:
                    return
                else:
                    node.doc = node.doc[1:]


        if self.compact:
            if node.arguments != []:
                argl = '(' + ','.join([x.name for x in node.arguments]) + ')'
            else:
                argl = ''


            d_ent = r"Function \texttt{" + node.name + argl + "} --- " + node.ret_val.type

            for a in node.ret_val.attributes:
                d_ent = d_ent + ", " + a

            self.print_line(r"\item[" + d_ent + r"]\mbox{} \par\noindent")

            if node.arguments != []:

                keys, rev_dict, func_args = combine_elements(node.arguments)

                self.print_line(r'\begin{description}')
                for k in keys:
                    for a in rev_dict[k][0]:
                        self.visit(a)
                self.print_line(r'\end{description}')

            for a in node.doc:
                self.print_line(a)
            self.print_line('')

            return

#        self.print_line( r"""\begin{center}\rule{10cm}{0.5pt}\end{center}""")
#        self.print_line( r"""\rule{\textwidth}{0.5pt}""")


        self.print_line(r'\index{general}{' + node.name + ' function}')

        if node.arguments != []:
            argl = '('
            for a in range(len(node.arguments)):
                arg = node.arguments[a]
                if isinstance(arg, ft.Declaration) and 'optional' in arg.attributes:
                    if argl[-2:] == '],':
                        argl = argl[:-2] + ',' + arg.name.rstrip() + '],'
                    elif argl.rstrip()[-4:] == '], &':
                        argl = argl.rstrip()[:-4] + ', &\n                        ' + arg.name.rstrip() + '],'
                    elif argl[-1] == ',':
                        argl = argl[:-1] + '[,' + arg.name.rstrip() + '],'
                    else:
                        argl = argl + '[' + arg.name.rstrip() + '],'
                else:
                    argl = argl + arg.name.rstrip() + ','
                if (a + 1) % 4 == 0.0 and a + 1 != len(node.arguments):
                    argl = argl + ' &\n                        '
            argl = argl[:-1] + ')'
        else:
            argl = ''

        if 'recursive' not in node.attributes:
            self.print_line(self.sections[self.depth] + r"""{Function \texttt{""" + node.name)
        else:
            self.print_line(self.sections[self.depth] + r"""{Recursive function\texttt{""" + node.name)
        if self.depth == 0:
            self.print_line("} (in file " + node.filename + ")}")
        else:
            self.print_line("""}}""")
#        self.print_line( self.sections[depth+1]+r'{Usage}')
#        self.print_line(r'\begin{boxedminipage}{\textwidth}')
        ret_name = node.ret_val.name
        if ret_name.lower() == node.name.lower():
            ret_name = ret_name[0].lower()
        self.print_line('>    ' + ret_name + ' = ' + node.name + argl)
#        self.print_line(r'\end{boxedminipage}'+'\n\n')
        for a in node.doc:
            self.print_line(a)

        if node.uses != []:
            self.print_line(self.sections[self.depth + 1] + r"{Uses}")
            u_temp = ''
            for a in node.uses:
                u_temp = u_temp + a + ', '
                if len(u_temp) > 50:
                    self.print_line(r"\texttt{" + u_temp[:-2] + "}")
                    u_temp = ''
                    self.print_line("\n")
            if u_temp != '':
                self.print_line(r"\texttt{" + u_temp[:-2] + "}")

        self.print_line(r"\begin{description}")

        if node.arguments != []:

            keys, rev_dict, func_args = combine_elements(node.arguments)

            for k in keys:
                for a in rev_dict[k][0]:
                    self.visit(a)

            for f in func_args:
                self.compact = True
                self.visit(f)
                self.compact = False

        #        self.print_line(self.sections[depth+1]+"{Return value --- ",)


        self.print_line(r"\item[Return value --- ",)

        self.is_ret_val = True
        self.visit(node.ret_val)
        self.is_ret_val = False

        self.print_line(r"]\mbox{} \par\noindent")
        for a in node.ret_val_doc:
            self.print_line(a)

        self.print_line(r"\end{description}")


    def visit_Declaration(self, node):

        if node.doc:
            if node.doc[0].strip() == 'OMIT':
                return

            if node.doc[0].strip() == 'OMIT SHORT':
                if self.short_doc:
                    return
                else:
                    node.doc = node.doc[1:]

        if self.is_ret_val:
            d_ent = node.type
            for a in node.attributes:
                d_ent = d_ent + ", " + a
            if node.value != '':
                d_ent = d_ent + r", value = \texttt{" + node.value + '}'
            self.print_line(d_ent)
            return

        if type(node.type) == type([]) and len(node.type) > 1:
            d_ent = r'\texttt{' + node.name + '} --- '


            for a in node.attributes:
                d_ent = d_ent + ' ' + a + ', '

            if d_ent[-1] == ',':
                d_ent = d_ent[:-2]  # remove trailing ','

            if (sum([len(t) for t in node.type]) + len(node.attributes) < 30):
                self.print_line(r"\item[" + d_ent + ' \emph{or} '.join(node.type) + r"]\mbox{} \par\noindent")
            else:
                self.print_line(r"\item[" + d_ent + r"]\mbox{} \par\noindent")
                self.print_line(r'\bfseries{' + ' \emph{or} '.join(node.type) + r'} \par\noindent')

        else:
            if (type(node.type) == type([])):
                typename = node.type[0]
            else:
                typename = node.type
            d_ent = r"\texttt{" + node.name + "} --- " + typename

            for a in node.attributes:
                d_ent = d_ent + ", " + a

            self.print_line(r"\item[" + d_ent + r"]\mbox{} \par\noindent")


#        if node.value!='':
#            d_ent=d_ent+r", value = \texttt{"+latex_escape(node.value)+'}'

        for a in node.doc:
            self.print_line(a)
        self.print_line('')



    def visit_Type(self, node):

        if node.doc:
            if node.doc[0].strip() == 'OMIT':
                return

            if node.doc[0].strip() == 'OMIT SHORT':
                if short_doc:
                    return
                else:
                    node.doc = node.doc[1:]


        #        self.print_line(r"""\begin{center}\rule{10cm}{0.5pt}\end{center}""")
        #        self.print_line( r"""\rule{\textwidth}{0.5pt}""")


        self.print_line(r'\index{general}{' + node.name + ' type}')

        self.print_line(r"""\subsection*{Type \texttt{""" + node.name + """}}""")
        for a in node.doc:
            self.print_line(a)

        self.print_line(r"""\subsubsection*{Elements}""")

        keys, rev_dict, func_args = combine_elements(node.elements)
        self.print_line(r"\begin{description}")

        for k in keys:
            for a in rev_dict[k][0]:
                self.visit(a)

        self.print_line(r"\end{description}")


    def visit_Interface(self, node):

        if node.doc:
            if node.doc[0].strip() == 'OMIT':
                return

            if node.doc[0].strip() == 'OMIT SHORT':
                if self.short_doc:
                    return
                else:
                    node.doc = node.doc[1:]



        #        self.print_line( r"""\rule{\textwidth}{0.5pt}""")

        if len(node.procedures) == 0:
            return

        if any([isinstance(proc, ft.Prototype) for proc in node.procedures]):
            log.debug('Skipping interface %s as some procedures were not found' % node.name)
            return

        n_sub = sum([isinstance(proc, ft.Subroutine) for proc in node.procedures])
        n_func = sum([isinstance(proc, ft.Function) for proc in node.procedures])

        if n_sub == len(node.procedures):
            is_sub = True
        elif n_func == len(node.procedures):
            is_sub = False
        else:
            raise ValueError('mixture of subroutines and functions in interface %s' % node.name)

        self.print_line(r'\index{general}{' + node.name + ' interface}')

        self.print_line(self.sections[self.depth] + r'{Interface \texttt{' + node.name + '}}')

        #        self.print_line(self.sections[depth+1]+r"""{Usage}""")

        printed_args = []
        #        self.print_line(r'\begin{boxedminipage}{\textwidth}')
        for sub in node.procedures:

            if sub.arguments != []:
                argl = '('
                for a in range(len(sub.arguments)):
                    arg = sub.arguments[a]
                    if isinstance(arg, ft.Declaration) and 'optional' in arg.attributes:
                        if argl[-2:] == '],':
                            argl = argl[:-2] + ',' + arg.name.rstrip() + '],'
                        elif argl.rstrip()[-4:] == '], &':
                            argl = argl.rstrip()[:-4] + ', &\n                        ' + arg.name.rstrip() + '],'
                        elif argl[-1] == ',':
                            argl = argl[:-1] + '[,' + arg.name.rstrip() + '],'
                        else:
                            argl = argl + '[' + arg.name.rstrip() + '],'
                    else:
                        argl = argl + arg.name.rstrip() + ','
                    if (a + 1) % 4 == 0.0 and a + 1 != len(sub.arguments):
                        argl = argl + ' &\n                        '
                argl = argl[:-1] + ')'
            else:
                argl = ''

            if not is_sub and sub.ret_val.name != sub.name:
                hash_value = argl
            else:
                hash_value = argl

            if hash_value in printed_args:
                continue

            printed_args.append(hash_value)

            if not is_sub:
                ret_name = sub.ret_val.name
                if ret_name.lower() == node.name.lower() or ret_name.lower() == sub.name.lower():
                    ret_name = ret_name[0].lower() + str(node.procedures.index(sub) + 1)
                self.print_line('>    ' + ret_name + ' = ' + node.name + argl)
            else:
                self.print_line('>    call ' + node.name + argl)
                #        self.print_line(r'\end{boxedminipage}'+'\n\n')


        for a in node.doc:
            self.print_line(a)

        for sub in node.procedures:
            for a in sub.doc:
                self.print_line(a)
            self.print_line('\n\n')

        got_args = (is_sub and sum([len(x.arguments) for x in node.procedures]) != 0) or not is_sub

        func_args = []
        if got_args:
            self.print_line(r'\begin{description}')


            arg_dict = {}
            i = 0  # counter for appearance order of args
            for sub in node.procedures:
                for a in sub.arguments:
                    if isinstance(a, ft.Subroutine) or isinstance(a, Function):
                        func_args.append(a)
                        continue
                    i = i + 1
                    if arg_dict.has_key(a.name):
                        if a.type.lower() + str(sorted(map(string.lower, a.attributes))) in \
                           [x[0].type.lower() + str(sorted(map(string.lower, x[0].attributes))) for x in arg_dict[a.name]]:
                            pass  # already got this name/type/attribute combo
                        else:
                            arg_dict[a.name].append((a, i))

                    else:
                        arg_dict[a.name] = [(a, i)]

            # Combine multiple types with the same name
            for name in arg_dict:
                types = [x[0].type for x in arg_dict[name]]
                types = uniq(types, string.lower)
                attr_lists = [x[0].attributes for x in arg_dict[name]]
                attributes = []

                contains_dimension = [ len([x for x in y if x.find('dimension') != -1]) != 0 for y in attr_lists ]

                for t in attr_lists:
                    attributes.extend(t)
                attributes = uniq(attributes, string.lower)

                dims = [x for x in attributes if x.find('dimension') != -1]
                attributes = [x for x in attributes if x.find('dimension') == -1]

                # If some attribute lists contains 'dimension' and some don't then
                # there are scalars in there as well.
                if True in contains_dimension and False in contains_dimension:
                    dims.insert(0, 'scalar')


                if (len(dims) != 0):
                    attributes.append(' \emph{or} '.join(dims))

                a = arg_dict[name][0][0]
                a.type = types  # r' \emph{or} '.join(types)
                a.attributes = attributes
                arg_dict[name] = (a, arg_dict[name][0][1])


            # Combine names with the same type, attributes and doc string
            rev_dict = {}
            for type, name in zip([ str([y.lower for y in x[0].type]) + \
                                     str([y.lower for y in x[0].attributes]) + str(x[0].doc) \
                                     for x in arg_dict.values() ], arg_dict.keys()):
                if rev_dict.has_key(type):
                    rev_dict[type].append(arg_dict[name])
                else:
                    rev_dict[type] = [arg_dict[name]]

            for k in rev_dict:
                names = [x[0].name for x in rev_dict[k]]
                a = rev_dict[k][0][0]
                names.sort(key=lambda x: arg_dict[x][1])

                # Split into pieces of max length 30 chars
                alist = []
                while names:
                    n = 0
                    length = 0
                    while (length < 30 and n < len(names)):
                        length = length + len(names[n])
                        n = n + 1
                    ns = names[:n]
                    del names[:n]
                    b = copy.copy(a)
                    b.name = ', '.join(ns)
                    alist.append(b)

                rev_dict[k] = (alist, min([x[1] for x in rev_dict[k]]))

            # Sort by original appearance order of first name
            keys = rev_dict.keys()
            keys.sort(key=lambda x: rev_dict[x][1])

            for k in keys:
                for a in rev_dict[k][0]:
                    self.visit(a)

            for f in func_args:
                self.compact = True
                self.visit(f)
                self.compact = False


        if not is_sub:
            #            self.print_line(self.sections[depth+1]+"{Return value --- ",)

            ret_types = [a.ret_val.type + str(a.ret_val.attributes) for a in node.procedures]

            if len(filter(lambda x: x != node.procedures[0].ret_val.type + str(node.procedures[0].ret_val.attributes), \
                          ret_types)) == 0:

                self.print_line(r"\item[Return value --- ",)
                self.is_ret_val = True
                self.visit(node.procedures[0].ret_val)
                self.is_ret_val = False
                self.print_line("]")
                for a in node.procedures[0].ret_val_doc:
                    self.print_line(a)
            else:
                self.print_line(r"\item[Return values:]\mbox{} \par\noindent")
                self.print_line(r'\begin{description}')
                for f in node.procedures:
                    shortname = f.ret_val.name[0].lower() + str(node.procedures.index(f) + 1)
                    self.print_line(r"\item[\texttt{" + shortname + "} --- ")
                    self.is_ret_val = True
                    self.visit(f.ret_val)
                    self.is_ret_val = False
                    self.print_line(']')
                    for a in f.ret_val_doc:
                        self.print_line(a)
                self.print_line(r'\end{description}')



        if got_args:
            self.print_line(r"\end{description}")


def write_latex(root, doc_title, doc_author, do_short_doc, intro, header, stream):
    # Print start
    if os.path.exists('COPYRIGHT'):
        for line in open('COPYRIGHT').readlines():
            stream.write('%' + line.strip() + '\n')

    if header:
        stream.write(r"""
\documentclass[11pt]{article}
\textheight 10in
\topmargin -0.5in
\textwidth 6.5in
\oddsidemargin -0.2in
\parindent=0.3in
\pagestyle{headings}

%Set depth of contents page
\setcounter{tocdepth}{2}

%\usepackage {makeidx, fancyhdr, boxedminipage, multind, colortbl, sverb}
\usepackage {makeidx, fancyhdr, colortbl, sverb}
\usepackage[dvips]{graphicx}
\pagestyle{fancy}

%\renewcommand{\sectionmark}[1]{\markboth{\thesection.\ #1}}
\renewcommand{\sectionmark}[1]{\markboth{}{#1}}
\fancyhf{}
\fancyhead[R]{\bfseries{\thepage}}
\fancyhead[L]{\bfseries{\rightmark}}
\renewcommand{\headrulewidth}{0.5pt}

%\makeindex{general}

\begin{document}


\title{""" + doc_title + r"""}
\date{\today}
\author{""" + doc_author + r"""}
\maketitle

\thispagestyle{empty}

\tableofcontents

% Defines paragraphs
\setlength{\parskip}{5mm}
\setlength{\parindent}{0em}

\newpage
""")

    if intro is not None:
        stream.write(r'\include{' + intro + '}')

    lg = LatexGenerator(stream, '', do_short_doc)

    for prog in root.programs:
        lg.visit(prog)

    for mod in root.modules:
        lg.visit(mod)

    if len(root.procedures) != 0:

        stream.write(r'\section{Miscellaneous Subroutines and Functions}' + '\n')

        for subt in root.procedures:
            lg.visit(subt)

    if header:
        stream.write(r"""
\printindex{general}{Index}

\end{document}

""")
