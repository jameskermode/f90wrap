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
import logging
import itertools
from f90wrap.fortran import *

# Define some regular expressions

module = re.compile('^module', re.IGNORECASE)
module_end = re.compile('^end\s*module', re.IGNORECASE)

program = re.compile('^program', re.IGNORECASE)
program_end = re.compile('^end\s*program', re.IGNORECASE)

attribs = r'allocatable|pointer|save|dimension *\(.*?\)|parameter|target|public|private'  # jrk33 added target

type_re = re.compile(r'^type((,\s*(' + attribs + r')\s*)*)(::)?\s*(?!\()', re.IGNORECASE)
type_end = re.compile('^end\s*type', re.IGNORECASE)

types = r'recursive|pure|double precision|elemental|(real\s*(\(.*?\))?)|(complex\s*(\(.*?\))?)|(integer\s*(\(.*?\))?)|(logical)|(character\s*(\(.*?\))?)|(type\s*\().*?(\))'
a_attribs = r'allocatable|pointer|save|dimension\(.*?\)|intent\(.*?\)|optional|target|public|private'

types_re = re.compile(types, re.IGNORECASE)

quoted = re.compile('(\".*?\")|(\'.*?\')')  # A quoted expression
comment = re.compile('!.*')  # A comment
whitespace = re.compile(r'^\s*')  # Initial whitespace
c_ret = re.compile(r'\r')

iface = re.compile('^interface', re.IGNORECASE)
iface_end = re.compile('^end\s*interface', re.IGNORECASE)

subt = re.compile(r'^(recursive\s+)?subroutine', re.IGNORECASE)
subt_end = re.compile(r'^end\s*subroutine\s*(\w*)', re.IGNORECASE)

recursive = re.compile('recursive', re.IGNORECASE)

funct = re.compile('^((' + types + r')\s+)*function', re.IGNORECASE)
# funct       = re.compile('^function',re.IGNORECASE)
funct_end = re.compile('^end\s*function\s*(\w*)', re.IGNORECASE)

prototype = re.compile(r'^module procedure ([a-zA-Z0-9_,\s]*)')

contains = re.compile('^contains', re.IGNORECASE)

uses = re.compile('^use\s+', re.IGNORECASE)
only = re.compile('only\s*:\s*', re.IGNORECASE)

decl = re.compile('^(' + types + r')\s*(,\s*(' + attribs + r')\s*)*(::)?\s*\w+(\s*,\s*\w+)*', re.IGNORECASE)
d_colon = re.compile('::')

attr_re = re.compile('(,\s*(' + attribs + r')\s*)+', re.IGNORECASE)
s_attrib_re = re.compile(attribs, re.IGNORECASE)


decl_a = re.compile('^(' + types + r')\s*(,\s*(' + a_attribs + r')\s*)*(::)?\s*\w+(\s*,\s*\w+)*', re.IGNORECASE)
attr_re_a = re.compile('(,\s*(' + a_attribs + r')\s*)+', re.IGNORECASE)
s_attrib_re_a = re.compile(a_attribs, re.IGNORECASE)

cont_line = re.compile('&')

fdoc_comm = re.compile(r'^!\s*\*FD')
fdoc_comm_mid = re.compile(r'!\s*\*FD')
fdoc_mark = re.compile('_FD\s*')
fdoc_rv_mark = re.compile('_FDRV\s*')

result_re = re.compile(r'result\s*\((.*?)\)', re.IGNORECASE)

arg_split = re.compile(r'\s*(\w*)\s*(\(.+?\))?\s*(=\s*[\w\.]+\s*)?,?\s*')

size_re = re.compile(r'size\(([^,]+),([^\)]+)\)')
dimension_re = re.compile(r'^([-0-9.e]+)|((rank\(.*\))|(size\(.*\))|(len\(.*\))|(slen\(.*\)))$')

alnum = string.ascii_letters + string.digits + '_'

valid_dim_re = re.compile(r'^(([-0-9.e]+)|(size\([_a-zA-Z0-9\+\-\*\/]*\))|(len\(.*\)))$')

public = re.compile('(^public$)|(^public\s*(\w+)\s*$)|(^public\s*::\s*(\w+)(\s*,\s*\w+)*$)', re.IGNORECASE)
private = re.compile('(^private$)|(^private\s*(\w+)\s*$)|(^private\s*::\s*(\w+)(\s*,\s*\w+)*$)', re.IGNORECASE)


def remove_delimited(line, d1, d2):

    bk = 0
    temp_str = ''
    undel_str = ''
    delimited = []

    for i in range(len(line)):
        if bk == 1:
            if line[i] == d2:
                bk = 0
                delimited.append(temp_str[:])
                temp_str = ''
                undel_str = undel_str + line[i]
                continue
            temp_str = temp_str + line[i]
            continue
        if line[i] == d1:
            bk = 1
        undel_str = undel_str + line[i]

    if bk == 1:
        undel_str = undel_str + temp_str

    return delimited, undel_str

def recover_delimited(line, d1, d2, delimited):

    if delimited == []:
        return line, []

    i = 0
    while i < len(line):
        if line[i] == d1:
            line = line[0:i + 1] + delimited[0] + line[i + 1:]
            i = i + len(delimited[0]) + 1
            delimited = delimited[1:]
        i = i + 1

    return line, delimited


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def split_attribs(atr):

    atr = atr.strip()
    if re.match('[,]', atr) != None:
        atr = atr[1:]
        atr = atr.strip()

    atrc = atr
    bk = 0
    atrl = []

    for i in range(len(atrc)):
        if atrc[i] == '(':
            bk = bk + 1
            if bk == 1:
                continue
        if atrc[i] == ')':
            bk = bk - 1
        if bk > 0:
            atrc = atrc[:i] + '0' + atrc[i + 1:]

    while re.search('[,]', atrc) != None:
        atrl.append(atr[:re.search('[,]', atrc).start()])  # jrk33 changed [\s,] to [,]
        atr = atr[re.search('[,]', atrc).end():]
        atrc = atrc[re.search('[,]', atrc).end():]

    if atr != '':
        atrl.append(atr)

    return list(map(lambda s : s.strip(), atrl))  # jrk33 added strip


hold_doc = None


class F90File(object):

    def __init__(self, fname):
        self.filename = fname
        self.file = open(fname, 'r')
        self.lines = self.file.readlines()
        self._lineno = 0
        self._lineno_offset = 0
        self.file.close()
        self.dquotes = []
        self.squotes = []

    @property
    def lineno(self):
        return self._lineno + self._lineno_offset

    def next(self):
        cline = ''

        while (cline == '' and len(self.lines) != 0):
            cline = self.lines[0].strip()
            if cline.find('_FD') == 1:
                break

            # jrk33 - join lines before removing delimiters

            # Join together continuation lines
            FD_index = cline.find('_FD')
            com2_index = cline.find('_COMMENT')
            if (FD_index == 0 or com2_index == 0):
                pass
            else:
                cont_index = cline.find('&')
                comm_index = cline.find('!')
                while (cont_index != -1 and (comm_index == -1 or comm_index > cont_index)):
                    cont = cline[:cont_index].strip()
                    cont2 = self.lines[1].strip()
                    if cont2.startswith('&'):
                        cont2 = cont2[1:]
                    cont = cont + cont2
                    self.lines = [cont] + self.lines[2:]
                    self._lineno = self._lineno + 1
                    cline = self.lines[0].strip()
                    cont_index = cline.find('&')

            # split by '!', if necessary
            comm_index = cline.find('!')
            if comm_index != -1:
                self.lines = [cline[:comm_index], cline[comm_index:]] + self.lines[1:]
                cline = self.lines[0]
                cline = cline.strip()
                # jrk33 - changed comment mark from '!*FD' to '!%'
                if self.lines[1].find('!%') != -1:
                    self.lines = [self.lines[0]] + ['_FD' + self.lines[1][2:]] + self.lines[2:]
                    self._lineno_offset = 1
                else:
                    self.lines = [self.lines[0]] + ['_COMMENT' + self.lines[1][1:]] + self.lines[2:]
                    self._lineno_offset = 1
            else:
                self._lineno_offset = 0
                self._lineno = self._lineno + 1

            self.lines = self.lines[1:]

        if cline == '':
            return None
        else:
            return cline



def check_uses(cline, file):

    if re.match(uses, cline) != None:
        cline = uses.sub('', cline)
        cline = cline.strip()
        out = re.match(re.compile(r"\w+"), cline).group()
        cline = file.next()
        return [out, cline]
    else:
        return [None, cline]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_doc(cline, file):

    if cline and re.match(fdoc_mark, cline) != None:
        out = fdoc_mark.sub('', cline)
        out = out.rstrip()
        cline = file.next()
        return [out, cline]
    else:
        return [None, cline]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_doc_rv(cline, file):

    cl = cline

    if cl is None:
        return [None, cl]

    if re.match(fdoc_rv_mark, cl) != None:
        out = fdoc_rv_mark.sub('', cl)
        out = out.rstrip()
        cl = file.next()
        return [out, cl]
    else:
        return [None, cl]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_cont(cline, file):

    cl = cline

    if re.match(contains, cl) != None:
        cl = file.next()
        return ['yes', cl]
    else:
        return [None, cl]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_program(cl, file):

    global hold_doc

    out = Program()
    cont = 0

    if re.match(program, cl) != None:
        # Get program name

        cl = program.sub('', cl)
        out.name = re.search(re.compile('\w+'), cl).group().strip()
        if out.name == '':
            out.name = '<Unnamed>'
        out.filename = file.filename
        out.lineno = file.lineno

        # Get next line, and check each possibility in turn
        cl = file.next()

        while re.match(program_end, cl) == None:

            # contains statement
            check = check_cont(cl, file)
            if check[0] != None:
                cont = 1
                cl = check[1]
                continue

            if cont == 0:

                # use statements
                check = check_uses(cl, file)
                if check[0] != None:
                    out.uses.append(check[0])
                    cl = check[1]
                    continue

                # Doc comment
                check = check_doc(cl, file)
                if check[0] != None:
                    out.doc.append(check[0])
                    cl = check[1]
                    continue
            else:

                # jrk33 - hold doc comment relating to next subrt or funct
                check = check_doc(cl, file)
                if check[0] != None:
                    if hold_doc == None:
                        hold_doc = [check[0]]
                    else:
                        hold_doc.append(check[0])
                    cl = check[1]
                    continue

                # Subroutine definition
                check = check_subt(cl, file)
                if check[0] != None:
                    logging.debug('    program subroutine ' + check[0].name)
                    out.procedures.append(check[0])
                    cl = check[1]
                    continue

                # Function definition
                check = check_funct(cl, file)
                if check[0] != None:
                    logging.debug('    program function ' + check[0].name)
                    out.procedures.append(check[0])
                    cl = check[1]
                    continue




            # If no joy, get next line
            cl = file.next()

        cl = file.next()

        out.lineno = slice(out.lineno, file.lineno - 1)
        return [out, cl]
    else:
        return [None, cl]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_module(cl, file):

    global hold_doc

    out = Module()
    cont = 0

    if re.match(module, cl) != None:

        out.filename = file.filename
        out.lineno = file.lineno

        # jrk33 - if we're holding a doc comment from before
        # subroutine definition, spit it out now
        if hold_doc is not None:
            for line in hold_doc:
                out.doc.append(line)
            hold_doc = None

        # Get module name
        cl = module.sub('', cl)
        out.name = re.search(re.compile('\w+'), cl).group()

        # Get next line, and check each possibility in turn

        cl = file.next()

        while re.match(module_end, cl) == None:

            # contains statement
            check = check_cont(cl, file)
            if check[0] != None:
                cont = 1
                cl = check[1]
                continue

            if cont == 0:

                # use statements
                check = check_uses(cl, file)
                if check[0] != None:
                    out.uses.append(check[0])
                    cl = check[1]
                    continue

                # Doc comment
                check = check_doc(cl, file)
                if check[0] != None:
                    if hold_doc == None:
                        hold_doc = [check[0]]
                    else:
                        hold_doc.append(check[0])
                    cl = check[1]
                    continue

                # jrk33 - Interface definition
                check = check_interface(cl, file)
                if check[0] != None:
                    logging.debug('    interface ' + check[0].name)
                    out.interfaces.append(check[0])
                    cl = check[1]
                    continue

                # Type definition
                check = check_type(cl, file)
                if check[0] != None:
                    logging.debug('    type ' + check[0].name)
                    out.types.append(check[0])
                    cl = check[1]
                    continue

                # Module variable
                check = check_decl(cl, file)
                if check[0] != None:
                    for a in check[0]:
                        out.elements.append(a)
                        cl = check[1]
                    continue

                # public and private access specifiers
                m = public.match(cl)
                if m is not None:
                    line = m.group()
                    if line.lower() == 'public':
                        logging.info('marking module %s as default public' % out.name)
                        out.default_access = 'public'
                    else:
                        line = line.lower().replace('public', '')
                        line = line.replace('::', '')
                        line = line.strip()
                        out.public_symbols.extend([field.strip() for field in line.split(',')])

                m = private.match(cl)
                if m is not None:
                    line = m.group()
                    if line.lower() == 'private':
                        logging.info('marking module %s as default private' % out.name)
                        out.default_access = 'private'
                    else:
                        line = line.replace('private', '')
                        line = line.replace('::', '')
                        line = line.strip()
                        out.private_symbols.extend([field.strip() for field in line.split(',')])

            else:

                # jrk33 - hold doc comment relating to next subrt or funct
                check = check_doc(cl, file)
                if check[0] != None:
                    if hold_doc == None:
                        hold_doc = [check[0]]
                    else:
                        hold_doc.append(check[0])
                    cl = check[1]
                    continue

                # Subroutine definition
                check = check_subt(cl, file)
                if check[0] != None:

                    logging.debug('    module subroutine ' + check[0].name)

                    for i in out.interfaces:
                        for j, p in enumerate(i.procedures):
                            if check[0].name.lower() == p.name.lower():
                                # Replace prototype with procedure
                                i.procedures[j] = check[0]
                                break
                        else:
                            continue
                        break
                    else:
                        out.procedures.append(check[0])

                    cl = check[1]
                    continue

                # Function definition
                check = check_funct(cl, file)
                if check[0] != None:

                    logging.debug('    module function ' + check[0].name)
                    for i in out.interfaces:
                        for j, p in enumerate(i.procedures):
                            if check[0].name.lower() == p.name.lower():
                                # Replace prototype with procedure
                                i.procedures[j] = check[0]
                                break
                        else:
                            continue
                        break
                    else:
                        out.procedures.append(check[0])
                    cl = check[1]
                    continue


            # If no joy, get next line
            cl = file.next()

        cl = file.next()

        out.lineno = slice(out.lineno, file.lineno - 1)
        return [out, cl]
    else:
        return [None, cl]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_subt(cl, file, grab_hold_doc=True):

    global hold_doc

    out = Subroutine()

    if re.match(subt, cl) != None:

        out.filename = file.filename
        out.lineno = file.lineno

        # Check if recursive

        if re.match(recursive, cl) != None:
            out.attributes.append('recursive')

        # Get subt name

        cl = subt.sub('', cl)
        out.name = re.search(re.compile('\w+'), cl).group()

        # Check to see if there are any arguments

        has_args = 0
        if re.search(r'\(.+', cl) != None:
            has_args = 1

        in_block_doc = False
        had_block_doc = False

        # get argument list

        if has_args:

            cl = re.sub('\w+', '', cl, count=1)
            argl = re.split('[\W]+', cl)

            del(argl[0])
            del(argl[len(argl) - 1])

            while cl.strip() == '' or re.search('&', cl) != None:
                cl = file.next()
                if cl.strip() == '': continue
                arglt = re.split('[\W]+', cl)
                del(arglt[len(arglt) - 1])
                for a in arglt:
                    argl.append()

        else:
            argl = []

        argl = list(map(lambda s : s.lower(), argl))

        # Get next line, and check each possibility in turn

        cl = file.next()

        while True:

            # Use statement
            # #check=check_uses(cl,file)
            # #if check[0]!=None:
            # #    out.uses.append(check[0])
            # #    cl=check[1]
            # #    continue

            # Look for block comments starting with a line of ======= or -------
            if cl is not None and not in_block_doc and not had_block_doc:
               if cl.startswith('_COMMENT=====') or cl.startswith('_COMMENT-----'):
                    in_block_doc = True

            if cl is not None and in_block_doc:
                if not cl.startswith('_COMMENT'):
                    in_block_doc = False
                    had_block_doc = True
                else:
                    rep = cl.strip().replace('_COMMENT', '')
                    if rep:
                        out.doc.append(rep)
                    cl = file.next()
                    continue

            # Doc comment
            check = check_doc(cl, file)
            if check[0] != None:
                out.doc.append(check[0])
                cl = check[1]
                continue

            if has_args:
                # Argument
                check = check_arg(cl, file)
                if check[0] != None:
                    for a in check[0]:
                        out.arguments.append(a)
                    cl = check[1]
                    continue

                # Interface section
                check = check_interface_decl(cl, file)
                if check[0] != None:
                    for a in check[0].procedures:
                        out.arguments.append(a)
                    cl = check[1]
                    continue

            m = subt_end.match(cl)

            if m == None:
                cl = file.next()
                continue
            elif m.group(1).lower() == out.name.lower() or m.group(1) == '':
                break

            # If no joy, get next line
            cl = file.next()



        # Select only first declaration that matches entries
        # in argument list

        if has_args:
            # t_re_str='(^'
            ag_temp = []
            # for a in argl:
            #    t_re_str=t_re_str+a+'$)|(^'
            # t_re_str=t_re_str[:-3]
            # t_re=re.compile(t_re_str,re.IGNORECASE)

            for i in out.arguments:
                if (i.name.lower() in argl and
                    len([a for a in ag_temp if a.name.lower() == i.name.lower()]) == 0):
                    ag_temp.append(i)

            out.arguments = ag_temp
            out.arguments.sort(key=lambda x:argl.index(x.name.lower()))

        else:
            out.arguments = []

        cl = file.next()

        # jrk33 - if we're holding a doc comment from before
        # subroutine definition, spit it out now
        if grab_hold_doc and hold_doc is not None:
            for line in hold_doc:
                out.doc.append(line)
            hold_doc = None


        out.lineno = slice(out.lineno, file.lineno - 1)
        return [out, cl]
    else:
        return [None, cl]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_funct(cl, file, grab_hold_doc=True):

    global hold_doc

    out = Function()

    if re.match(funct, cl) != None:

        out.filename = file.filename
        out.lineno = file.lineno
        out.ret_val.filename = out.filename
        out.ret_val.lineno = out.lineno

        # Check if recursive

        if re.search(recursive, cl) != None:
            out.attributes.append('recursive')
            cl = recursive.sub('', cl)

        # Get return type, if present

        cl = cl.strip()
        if re.match(types_re, cl) != None:
            out.ret_val.type = re.match(types_re, cl).group()


        # jrk33 - Does function header specify alternate name of
        # return variable?
        ret_var = None
        if re.search(result_re, cl) != None:
            ret_var = re.search(result_re, cl).group(1)
            cl = result_re.sub('', cl)

        # Get func name

        cl = funct.sub('', cl)
        out.name = re.search(re.compile('\w+'), cl).group()

        # Default name of return value is function name
        out.ret_val.name = out.name



        # Check to see if there are any arguments

        if re.search(r'\([^\)]+', cl) != None:
            has_args = 1
        else:
            has_args = 0

        if has_args:
            # get argument list

            cl = re.sub('\w+', '', cl, count=1)
            argl = re.split('[\W]+', cl)

            del(argl[0])
            del(argl[len(argl) - 1])

            while cl.strip() == '' or re.search('&', cl) != None:
                cl = file.next()
                if cl.strip() == '':
                    continue
                arglt = re.split('[\W]+', cl)
                del(arglt[len(arglt) - 1])
                for a in arglt:
                    argl.append(a.lower())
        else:
            argl = []

        argl = list(map(lambda s : s.lower(), argl))

        # Get next line, and check each possibility in turn

        in_block_doc = False
        had_block_doc = False

        cl = file.next()


        while True:

            # Use statement
            # #check=check_uses(cl,file)
            # #if check[0]!=None:
            # #    out.uses.append(check[0])
            # #    cl=check[1]
            # #    continue

            # Look for block comments starting with a line of ======= or -------
            if cl is not None and not in_block_doc and not had_block_doc:
               if cl.startswith('_COMMENT=====') or cl.startswith('_COMMENT-----'):
                    in_block_doc = True

            if cl is not None and in_block_doc:
                if not cl.startswith('_COMMENT'):
                    in_block_doc = False
                    had_block_doc = True
                else:
                    rep = cl.strip().replace('_COMMENT', '')
                    if rep:
                        out.doc.append(rep)
                    cl = file.next()
                    continue

            # Doc comment - return value
            check = check_doc_rv(cl, file)
            if check[0] != None:
                out.ret_val_doc.append(check[0])
                cl = check[1]
                continue

            # Doc comment
            check = check_doc(cl, file)
            if check[0] != None:
                out.doc.append(check[0])
                cl = check[1]
                continue

            # Interface section
            check = check_interface_decl(cl, file)
            if check[0] != None:
                for a in check[0].procedures:
                    out.arguments.append(a)
                cl = check[1]
                continue

            # Argument
            check = check_arg(cl, file)
            if check[0] != None:
                for a in check[0]:
                    out.arguments.append(a)
                    cl = check[1]
                continue

            m = re.match(funct_end, cl)

            if m == None:
                cl = file.next()
                continue

            elif m.group(1).lower() == out.name.lower() or m.group(1) == '':
                break

            cl = file.next()

        # Select only first declaration that matches entries
        # in argument list

        ag_temp = []

        # if has_args:
        #    t_re_str='(^'
        #    for a in argl:
        #        t_re_str=t_re_str+a+'$)|(^'
        #   t_re_str=t_re_str[:-3]
        #    t_re=re.compile(t_re_str,re.IGNORECASE)

        name_re = re.compile(out.name, re.IGNORECASE)

        for i in out.arguments:
            if has_args and i.name.lower() in argl and \
                   len([a for a in ag_temp if a.name.lower() == i.name.lower()]) == 0:
                ag_temp.append(i)
            if re.search(name_re, i.name) != None:
                out.ret_val = i
            if ret_var != None and i.name.lower().strip() == ret_var.lower().strip():
                out.ret_val = i

        out.arguments = ag_temp
        out.arguments.sort(key=lambda x:argl.index(x.name.lower()))

        cl = file.next()

        # jrk33 - if we're holding a doc comment from before
        # subroutine definition, spit it out now
        if grab_hold_doc and hold_doc is not None:
            for line in hold_doc:
                out.doc.append(line)
            hold_doc = None

        out.lineno = slice(out.lineno, file.lineno - 1)
        return [out, cl]
    else:
        return [None, cl]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_type(cl, file):

#    global hold_doc

    out = Type()
    m = re.match(type_re, cl)
    current_access = None

    if m is not None:

        out.filename = file.filename
        out.lineno = file.lineno

        # jrk33 - see if it's a global variable of this type.
        # if so, do nothing - it will be found by check_decl
        if decl.match(cl) != None:
            return [None, cl]

#        if hold_doc != None:
#            for line in hold_doc:
#                out.doc.append(line)
#            hold_doc = None


        # Get type name
        cl = type_re.sub('', cl)

        # Check if there are any type attributes
        out.attributes = []
        if m.group(1):
            out.attributes = split_attribs(m.group(1))

        out.name = re.search(re.compile('\w+'), cl).group()
        logging.info('parser reading type %s' % out.name)

        # Get next line, and check each possibility in turn

        cl = file.next()

        while re.match(type_end, cl) == None:
            check = check_doc(cl, file)
            if check[0] != None:
                out.doc.append(check[0])
                cl = check[1]
                continue

            check = check_decl(cl, file)
            if check[0] != None:
                for a in check[0]:
                    if current_access is not None:
                        a.attributes.append(current_access)
                    out.elements.append(a)
                cl = check[1]
                continue

            if cl.lower() == 'public':
                current_access = 'public'

            elif cl.lower() == 'private':
                current_access = 'private'

            cl = file.next()

        cl = file.next()

        out.lineno = slice(out.lineno, file.lineno - 1)
        return [out, cl]
    else:
        return [None, cl]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_interface(cl, file):

    global hold_doc

    out = Interface()

    if re.match(iface, cl) != None:

        out.filename = file.filename
        out.lineno = file.lineno

        cl = iface.sub('', cl)
        out.name = cl.strip()

        # if out.name == '':
        #    return [None, cl]

        if hold_doc is not None:
            for line in hold_doc:
                out.doc.append(line)
            hold_doc = None

        cl = file.next()
        while re.match(iface_end, cl) == None:

            check = check_doc(cl, file)
            if check[0] != None:
                out.doc.append(check[0])
                cl = check[1]
                continue

            check = check_prototype(cl, file)
            if check[0] != None:
                for a in check[0]:
                    out.procedures.append(a)
                cl = check[1]
                continue

            cl = file.next()

        cl = file.next()

        out.lineno = slice(out.lineno, file.lineno - 1)
        return [out, cl]

    else:
        return [None, cl]


def check_interface_decl(cl, file):

    out = Interface()

    if cl and re.match(iface, cl) != None:

        out.filename = file.filename
        out.lineno = file.lineno

        cl = file.next()
        while re.match(iface_end, cl) == None:

            # Subroutine declaration
            check = check_subt(cl, file, grab_hold_doc=False)
            if check[0] != None:
                out.procedures.append(check[0])
                cl = check[1]
                continue

            # Function declaration
            check = check_funct(cl, file, grab_hold_doc=False)
            if check[0] != None:
                out.procedures.append(check[0])
                cl = check[1]
                continue

            cl = file.next()

        cl = file.next()

        out.lineno = slice(out.lineno, file.lineno - 1)
        return [out, cl]

    else:
        return [None, cl]


def check_prototype(cl, file):

    m = prototype.match(cl)
    if m != None:
        out = map(lambda s : s.strip().lower(), m.group(1).split(','))
        out = [Prototype(name=name, lineno=file.lineno, filename=file.filename) for name in out]

        cl = file.next()
        return [out, cl]

    else:
        return [None, cl]





#+++++++++++++++++++++++++++++++++++++++++++++++++++++++


def check_decl(cl, file):

    out = []

    if re.match(decl, cl) != None:

        filename = file.filename
        lineno = file.lineno

        tp = re.match(types_re, cl).group()
        atr = re.search(attr_re, cl)
        if atr != None:
            atrl = s_attrib_re.findall(atr.group())
            for j in range(len(atrl)):
                atrl[j] = atrl[j].rstrip()
        else:
            atrl = []
        m = re.search(d_colon, cl)
        if m is not None:
            names = cl[m.end():]
        else:
            names = types_re.sub('', cl)

        # old line - doesn't handle array constants
        # nl=re.split(r'\s*,\s*',names)
        nl = split_attribs(names)


        alist = []
        for j in range(len(atrl)):
            alist.append(atrl[j])

        cl = file.next()
        check = check_doc(cl, file)

        dc = []
        while check[0] != None:
            # Doc comment
            dc.append(check[0])
            cl = check[1]
            check = check_doc(cl, file)

        cl = check[1]

        for i in range(len(nl)):
            nl[i] = nl[i].strip()
            nlv = re.split(r'\s*=\s*', nl[i])

            names, sizes = splitnames(nlv[0])
            temp = Element(name=names[0], type=tp, doc=dc, attributes=alist[:],
                         filename=filename, lineno=lineno)
            if len(nlv) == 2:
                temp.value = nlv[1]
            if sizes[0] != '':
                temp.attributes.append('dimension' + sizes[0])
            out.append(temp)

        return [out, cl]
    else:
        return [None, cl]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def splitnames(names):
   nl = []
   sizes = []
   b = 0
   namestart = 0
   sizestart = 0
   name = ''
   size = ''
   for i, n in enumerate(names):
      if n == '(':
         b += 1
         size += '('
      elif n == ')':
         b -= 1
         size += ')'
      elif n == ',' and b == 0:
         nl.append(name)
         name = ''
         sizes.append(size)
         size = ''
      elif b == 0:
         name += n
      else:
         size += n

   nl.append(name)
   sizes.append(size)

   return nl, sizes


def check_arg(cl, file):

    out = []

    if cl and re.match(decl_a, cl) != None:

        filename = file.filename
        lineno = file.lineno


        tp = re.match(types_re, cl).group()
        m = re.search(d_colon, cl)
        if m is not None:
            atr_temp = cl[re.match(types_re, cl).end():m.start()]
            names = cl[m.end():]
        else:
            atr_temp = ''
            # Need to remove ONLY THE FIRST type string (the name may have the type in it)
            names = types_re.sub('', cl, 1)


        atrl = split_attribs(atr_temp)

#        names=cl[re.search(d_colon,cl).end():]
# #        nl=re.split(',',names)
# #        for i in range(len(nl)):
# #            nl[i]=nl[i].strip()


        # jrk33 - added code to cope with array declarations with
        # size after variable name, e.g. matrix(3,3) etc.

        # Remove values
        names = re.sub(r'=.*$', '', names)

        nl, sizes = splitnames(names)


        alist = []
        for j in range(len(atrl)):
            alist.append(atrl[j])

        cl = file.next()
        check = check_doc(cl, file)

        dc = []

        while check[0] != None:
            # Doc comment
            dc.append(check[0])
            cl = check[1]
            check = check_doc(cl, file)

        cl = check[1]

        for i in range(len(nl)):
            nl[i] = nl[i].strip()
            temp = Argument(name=nl[i], doc=dc, type=tp, attributes=alist[:],
                          filename=filename, lineno=lineno)

            # Append dimension if necessary
            if sizes[i] != '':
                temp.attributes.append('dimension' + sizes[i])

            out.append(temp)

        return [out, cl]
    else:
        return [None, cl]


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++


def read_files(args):

    global hold_doc

    root = Root()

    for fn in args:

        fname = fn

        # Open the filename for reading

        logging.debug('processing file ' + fname)
        file = F90File(fname)

        # Get first line

        cline = file.next()

        while cline != None:

            # programs
            check = check_program(cline, file)
            if check[0] != None:
                logging.debug('  program ' + check[0].name)
                root.programs.append(check[0])
                cline = check[1]
                continue

            # modules
            check = check_module(cline, file)
            if check[0] != None:
                logging.debug('  module ' + check[0].name)
                root.modules.append(check[0])
                cline = check[1]
                continue

            # jrk33 - hold doc comment relating to next module, subrt or funct
            check = check_doc(cline, file)
            if check[0] != None:
                if hold_doc == None:
                    hold_doc = [check[0]]
                else:
                    hold_doc.append(check[0])
                cline = check[1]
                continue


            # stand-alone subroutines
            check = check_subt(cline, file)
            if check[0] != None:
                logging.debug('  subroutine ' + check[0].name)
                root.procedures.append(check[0])
                cline = check[1]
                continue

            # stand-alone functions
            check = check_funct(cline, file)
            if check[0] != None:
                logging.debug('  function ' + check[0].name)
                root.procedures.append(check[0])
                cline = check[1]
                continue

            cline = file.next()

    # apply some rules to the parsed tree
    from f90wrap.fortran import fix_argument_attributes, LowerCaseConverter

    root = fix_argument_attributes(root)
    root = LowerCaseConverter().visit(root)
    return root
