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

import logging
import string
import sys
import os
import re
import json

from f90wrap.fortran import (Fortran, Root, Program, Module,
                             Procedure, Subroutine, Function, Interface,
                             Prototype, Declaration,
                             Argument, Element, Type,
                             fix_argument_attributes,
                             LowerCaseConverter,
                             RepeatedInterfaceCollapser,
                             Binding,)


log = logging.getLogger(__name__)


# Define some regular expressions

module = re.compile(r'^module|^submodule', re.IGNORECASE)
module_end = re.compile(r'^end\s*module|^end\s*submodule|end$', re.IGNORECASE)

submodule = re.compile(r'^submodule', re.IGNORECASE)

program = re.compile(r'^program', re.IGNORECASE)
program_end = re.compile(r'^end\s*program|end$', re.IGNORECASE)

attribs = r'allocatable|pointer|save|contiguous|dimension *\(.*?\)|parameter|target|public|private|abstract|extends *\(.*?\)|bind\(C\)|value'  # jrk33 added target, value added for issue #171

type_re = re.compile(r'^type((,\s*(' + attribs + r')\s*)*)(::)?\s*(?!\()', re.IGNORECASE)
type_end = re.compile(r'^end\s*type|end$', re.IGNORECASE)

dummy_types_re = re.compile(r'recursive|pure|elemental', re.IGNORECASE)

prefixes = r'elemental|impure|module|non_recursive|pure|recursive'
types = r'double precision|(real\s*(\(.*?\))?)|(complex\s*(\(.*?\))?)|(integer\s*(\(.*?\))?)|(logical)|(character\s*(\(.*?\))?)|(type\s*\().*?(\))|(class\s*\().*?(\))'
a_attribs = r'allocatable|pointer|save|dimension\(.*?\)|intent\(.*?\)|optional|target|public|private|contiguous|value'

types_re = re.compile(types, re.IGNORECASE)

quoted = re.compile(r'(\".*?\")|(\'.*?\')')  # A quoted expression
comment = re.compile(r'!.*')  # A comment
whitespace = re.compile(r'^\s*')  # Initial whitespace
c_ret = re.compile(r'\r')

iface = re.compile(r'^interface', re.IGNORECASE)
abstract_iface = re.compile(r'^abstract\s*interface', re.IGNORECASE)
iface_end = re.compile(r'^end\s*interface|end$', re.IGNORECASE)

subt = re.compile(r'^((' + prefixes + r')\s+)*subroutine', re.IGNORECASE)
subt_end = re.compile(r'^end\s*subroutine\s*(\w*)|end$', re.IGNORECASE)

funct = re.compile(r'^((' + types + '|' + prefixes + r')\s+)*function', re.IGNORECASE)
# funct       = re.compile(r'^function', re.IGNORECASE)
funct_end = re.compile(r'^end\s*function\s*(\w*)|end$', re.IGNORECASE)

prototype = re.compile(r'^module procedure\s*(::)?\s*([a-zA-Z0-9_,\s]*)', re.IGNORECASE)

binding_types = r'procedure|generic|final'
binding_type_group = r'^(' + binding_types + r')'
no_bracket_for_interface = r'(?!\s*\()'
double_dot_and_attributes = r'\s*((,([^:]*))?(::))?'
name_target_group = r'\s*(.*)'

binding = re.compile(
    binding_type_group +
    no_bracket_for_interface +
    double_dot_and_attributes +
    name_target_group,
    re.IGNORECASE
)

interface_in_brackets = r'\s*\((.+)\)'
deferred_binding = re.compile(
    r'^(procedure)' +
    interface_in_brackets +
    double_dot_and_attributes +
    name_target_group,
    re.IGNORECASE
)

contains = re.compile(r'^contains', re.IGNORECASE)

uses = re.compile(r'^use\s+', re.IGNORECASE)
only = re.compile(r'only\s*:\s*', re.IGNORECASE)

decl = re.compile(r'^(' + types + r')\s*(,\s*(' + attribs + r')\s*)*(::)?\s*\w+(\s*,\s*\w+)*', re.IGNORECASE)
d_colon = re.compile(r'::')

attr_re = re.compile(r'(,\s*(' + attribs + r')\s*)+', re.IGNORECASE)
s_attrib_re = re.compile(attribs, re.IGNORECASE)

decl_a = re.compile(r'^(' + types + r')\s*(,\s*(' + a_attribs + r')\s*)*(::)?\s*\w+(\s*,\s*\w+)*', re.IGNORECASE)
attr_re_a = re.compile(r'(,\s*(' + a_attribs + r')\s*)+', re.IGNORECASE)
s_attrib_re_a = re.compile(a_attribs, re.IGNORECASE)

cont_line = re.compile(r'&')

fdoc_comm = re.compile(r'^!\s*\*FD')
fdoc_comm_mid = re.compile(r'!\s*\*FD')
fdoc_mark = re.compile(r'_FD\s*')
fdoc_rv_mark = re.compile(r'_FDRV\s*')

doxygen_main = re.compile(r'_COMMENT.*\\(brief|details)')
doxygen_others = re.compile(r'_COMMENT.*\\(file|author|copyright)')
doxygen_param = re.compile(r'_COMMENT.*\\(param|returns)')
doxygen_param_group = re.compile(r'_COMMENT.*\\(param|returns)\s*(\[.+?\]|)\s+(\S+?)[\s:]+(.*)')
comment_pattern = re.compile(r'_COMMENT[<>]?')

result_re = re.compile(r'result\s*\((.*?)\)', re.IGNORECASE)

arg_split = re.compile(r'\s*(\w*)\s*(\(.+?\))?\s*(=\s*[\w\.]+\s*)?,?\s*')

size_re = re.compile(r'size\(([^,]+),([^\)]+)\)', re.IGNORECASE)
dimension_re = re.compile(r'^([-0-9.e]+)|((rank\(.*\))|(size\(.*\))|(len\(.*\))|(slen\(.*\)))$', re.IGNORECASE)

alnum = string.ascii_letters + string.digits + '_'

valid_dim_re = re.compile(r'^(([-0-9.e]+)|(size\([_a-zA-Z0-9\+\-\*\/]*\))|(len\(.*\)))$', re.IGNORECASE)

public = re.compile(r'(^public$)|(^public\s*(::)?\s*(\w+)(\s*,\s*\w+)*$)', re.IGNORECASE)
private = re.compile(r'(^private$)|(^private\s*(::)?\s*(\w+)(\s*,\s*\w+)*$)', re.IGNORECASE)

rmspace = re.compile(r'(\w+)\s+\(', re.IGNORECASE)
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


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def split_attribs(atr):
    atr = atr.strip()
    if re.match(r'[,]', atr) != None:
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

    while re.search(r'[,]', atrc) != None:
        atrl.append(atr[:re.search(r'[,]', atrc).start()])  # jrk33 changed [\s,] to [,]
        atr = atr[re.search(r'[,]', atrc).end():]
        atrc = atrc[re.search(r'[,]', atrc).end():]

    if atr != '':
        atrl.append(atr)

    return list(map(lambda s: s.strip(), atrl))  # jrk33 added strip


hold_doc = None


class F90File(object):
    def __init__(self, fname):
        self.filename = fname
        self.file = open(fname, 'r', encoding='utf-8', errors='ignore')
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
            while (cline == '' and len(self.lines) != 1): # issue105 - rm empty lines
                self.lines = self.lines[1:]
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
                try:
                    cont2 = self.lines[1].strip()
                    while (cont2 == '' and len(self.lines) != 2): # issue105 - rm empty lines
                        self.lines[1:] = self.lines[2:]
                        cont2 = self.lines[1].strip()
                    if cont2.startswith('&'):
                        cont2_index = 0
                    else:
                        cont2_index = -1
                except:
                    cont2_index = -1
                comm_index = cline.find('!')
                while (cont_index != -1 and (comm_index == -1 or comm_index > cont_index)) or \
                        (cont2_index != -1):
                    cont2 = self.lines[1].strip()
                    while (cont2 == '' and len(self.lines) != 2): # issue105 - rm empty lines
                        self.lines[1:] = self.lines[2:]
                        cont2 = self.lines[1].strip()
                    if cont2.startswith('&'):
                        cont2 = cont2[1:].strip()

                    # Skip interleaved comments starting with `!`
                    if cont_index != -1 and not cont2.startswith('!'):
                        cont = cline[:cont_index].strip()
                    else:
                        cont = cline.strip()
                    if not cont2.startswith('!'):
                        cont = cont + cont2

                    self.lines = [cont] + self.lines[2:]
                    self._lineno = self._lineno + 1
                    cline = self.lines[0].strip()
                    while (cline == '' and len(self.lines) != 1): # issue105 - rm empty lines
                        self.lines = self.lines[1:]
                        cline = self.lines[0].strip()
                    cont_index = cline.find('&')
                    try:
                        cont2 = self.lines[1].strip()
                        while (cont2 == '' and len(self.lines) != 2): # issue105 - rm empty lines
                            self.lines[1:] = self.lines[2:]
                            cont2 = self.lines[1].strip()
                        if cont2.startswith('&'):
                            cont2_index = 0
                        else:
                            cont2_index = -1
                    except:
                        cont2_index = -1
                    comm_index = cline.find('!')

            # split by '!', if necessary
            comm_index = cline.find('!')
            if comm_index != -1:
                self.lines = [cline[:comm_index], cline[comm_index:]] + self.lines[1:]
                cline = self.lines[0].strip()
                # jrk33 - changed comment mark from '!*FD' to '!%'
                if self.lines[1].find('!%') != -1:
                    self.lines = [self.lines[0]] + ['_FD' + self.lines[1][2:]] + self.lines[2:]
                    self._lineno_offset = 1
                else:
                    self.lines[1] = re.sub(r'^!+', '_COMMENT', self.lines[1])
                    self._lineno_offset = 1
            else:
                self._lineno_offset = 0
                self._lineno = self._lineno + 1

            self.lines = self.lines[1:]

        if not cline.startswith('_COMMENT'):
            cline = rmspace.sub(r'\1(', cline)
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


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_doc(cline, file):
    out = None
    if cline:
        for pattern in [fdoc_mark, doxygen_main, doxygen_others, doxygen_param]:
            match = re.search(pattern, cline)
            if match != None:
                # Init out, only doxygen_other tags need a prefix
                if pattern == doxygen_others:
                    out = match.group(1).capitalize() + ': '
                else:
                    out = ""
                stop = False
                # Ensure that the whole comment is captured even if it is multiline
                # Only parse 100 lines maximum to be sure not to enter a infinite loop
                for _ in range(100):
                    cleaned_line = pattern.sub('', cline)
                    cleaned_line = comment_pattern.sub('', cleaned_line)
                    cleaned_line = cleaned_line.strip(' ')
                    if pattern == doxygen_param:
                        # Leave pattern for later parsing in check_arg
                        out = cline.strip()
                        stop = True
                    elif pattern == doxygen_main:
                        out += cleaned_line + '\n'
                    elif pattern == doxygen_others:
                        out += cleaned_line
                    else:
                        out += cleaned_line
                    # Get next line and check if it is a multiline comment
                    cline = file.next()

                    # If the next line is not a comment, stop
                    if not re.search(comment_pattern, cline):
                        stop = True
                    # If the next line contains a doxygen tag, stop
                    for next_pattern in [fdoc_mark, doxygen_main, doxygen_others, doxygen_param]:
                        if re.search(next_pattern, cline):
                            stop = True
                            break
                    if stop:
                        break
                return [out, cline]
    return [out, cline]



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

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


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_cont(cline, file):
    cl = cline

    if re.match(contains, cl) != None:
        cl = file.next()
        return ['yes', cl]
    else:
        return [None, cl]


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_program(cl, file):
    global doc_plugin_module
    global hold_doc

    out = Program()
    cont = 0

    if re.match(program, cl) != None:
        # Get program name

        cl = program.sub('', cl)
        out.name = re.search(re.compile(r'\w+'), cl).group().strip()
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
                    log.debug('    program subroutine ' + check[0].name)
                    out.procedures.append(check[0])
                    cl = check[1]
                    continue

                # Function definition
                check = check_funct(cl, file)
                if check[0] != None:
                    log.debug('    program function ' + check[0].name)
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


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_module(cl, list_modules, file):
    global doc_plugin_module
    global hold_doc

    cont = 0

    if re.match(module, cl) != None:

        is_submodule = re.match(submodule, cl) != None

        # Get (sub)module name
        cl = module.sub('', cl)
        name = re.search(re.compile(r'\w+'), cl).group()

        # If module already registered, just append new informations
        # This can happen with submodules, different files are associated to the same module
        flag_exists = False
        for old_module in list_modules:
            if old_module.name == name:
                out = old_module
                flag_exists = True
                break

        if not flag_exists:
            out = Module()
            out.name = name

        # In case of submodules, keep name of main module only
        if not is_submodule:
            out.name = name
            out.filename = file.filename
            out.lineno = file.lineno

        # jrk33 - if we're holding a doc comment from before
        # subroutine definition, spit it out now
        if hold_doc is not None:
            for line in hold_doc:
                out.doc.append(line)
            hold_doc = None

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
                    log.debug('    interface ' + check[0].name)
                    check[0].mod_name = out.name
                    out.interfaces.append(check[0])
                    cl = check[1]
                    continue

                # Abstract interface definition
                check = check_abstract_interface(cl, file)
                if check[0] != None:
                    log.debug('    abstract interface ')
                    check[0].mod_name = out.name
                    for proc in check[0].procedures:
                        proc.mod_name = out.name
                    out.interfaces.append(check[0])
                    cl = check[1]
                    continue

                # Type definition
                check = check_type(cl, file)
                if check[0] != None:
                    log.debug('    type ' + check[0].name)
                    check[0].mod_name = out.name
                    out.types.append(check[0])
                    cl = check[1]
                    continue

                # Module variable
                check = check_decl(cl, file)
                if check[0] != None:
                    for el in check[0]:
                        # Some module variable have multiple declaration in different #ifdef scopes
                        if el.name not in [elem.name for elem in out.elements]:
                          out.elements.append(el)
                        cl = check[1]
                    continue

                # public and private access specifiers
                m = public.match(cl)
                if m is not None:
                    line = m.group()
                    if line.lower() == 'public':
                        log.info('marking module %s as default public' % out.name)
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
                        log.info('marking module %s as default private' % out.name)
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
                    log.debug('    module subroutine ' + check[0].name)
                    check[0].mod_name = out.name
                    out.procedures.append(check[0])
                    cl = check[1]
                    continue

                # Function definition
                check = check_funct(cl, file)
                if check[0] != None:
                    log.debug('    module function ' + check[0].name)
                    check[0].mod_name = out.name
                    out.procedures.append(check[0])
                    cl = check[1]
                    continue

            # If no joy, get next line
            cl = file.next()

        cl = file.next()

        if not flag_exists:
            out.lineno = slice(out.lineno, file.lineno - 1)
            list_modules.append(out)

        return [out, cl]
    else:
        return [None, cl]


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_subt(cl, file, grab_hold_doc=True):
    global doc_plugin_module
    global hold_doc

    out = Subroutine()

    if re.match(subt, cl) != None:

        out.filename = file.filename
        out.lineno = file.lineno

        # Check if recursive, elemental or pure
        m = re.match(dummy_types_re, cl)
        if m != None:
            out.attributes.append(m.group())

        # Get subt name
        cl = subt.sub('', cl)
        out.name = re.search(re.compile(r'\w+'), cl).group()
        log.debug('    module subroutine checking ' + out.name)

        # Test in principle whether we can have a 'do not wrap' list
        if out.name.lower() == 'debugtype_stop_if':
            return [None, cl]

        # Check to see if there are any arguments

        has_args = 0
        if re.search(r'\(.+', cl) != None:
            has_args = 1

        in_block_doc = False
        had_block_doc = False

        # get argument list

        if has_args and ')' in cl:
            cl = cl[:cl.find(')', 0)+1]
            cl = re.sub(r'\w+', '', cl, count=1)
            argl = re.split(r'[\W]+', cl)

            del (argl[0])
            del (argl[len(argl) - 1])

            while cl.strip() == '' or re.search(r'&', cl) != None:
                cl = file.next()
                if cl.startswith('_COMMENT'):
                    cl = file.next()
                if cl.strip() == '': continue
                arglt = re.split(r'[\W]+', cl)
                del (arglt[len(arglt) - 1])
                for a in arglt:
                    argl.append(a)

        else:
            argl = []

        argl = list(map(lambda s: s.lower(), argl))

        # Get next line, and check each possibility in turn

        cl = file.next()

        cont = 0
        subroutine_lines = []
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

            # contains statement
            check = check_cont(cl, file)
            if check[0] is not None:
                cont = 1
                cl = check[1]

            if cont == 0:

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

            else:

                while True :
                    # Subroutine definition
                    check = check_subt(cl, file)
                    if check[0] is not None:
                        # Discard contained subroutine
                        cl = check[1]
                        continue

                    # Function definition
                    check = check_funct(cl, file)
                    if check[0] is not None:
                        # Discard contained function
                        cl = check[1]
                        continue
                    break

            m = subt_end.match(cl)

            subroutine_lines.append(cl)
            if m == None:
                cl = file.next()
                continue
            else:
                if doc_plugin_module is not None:
                    extra_doc = doc_plugin_module.doc_plugin(subroutine_lines, out.name, 'subroutine')
                    out.doc.extend(extra_doc)
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

            implicit_to_explicit_arguments(argl, ag_temp)

            out.arguments = ag_temp
            out.arguments.sort(key=lambda x: argl.index(x.name.lower()))

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


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def implicit_to_explicit_arguments(argl, ag_temp):
    # YANN: Give a Type to undeclared arguments in the arguments list, following the implicit arguments type rule
    implicit_arguments = set(argl) - set(a.name.lower() for a in ag_temp)
    for i in implicit_arguments:
        ag_temp.append(
            Argument(name=i, doc=None, type=implicit_type_rule(i), attributes=None, filename=None, lineno=None))


def implicit_type_rule(var):
    # YANN: implicit arguments type rule
    tp = 'integer' if var[0] in ('i', 'j', 'k', 'l', 'm', 'n') else 'real'
    log.debug('        implicit type of "%s" inferred from its name as "%s"' % (var, tp))
    return tp


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_funct(cl, file, grab_hold_doc=True):
    global doc_plugin_module
    global hold_doc

    out = Function()

    if re.match(funct, cl) != None:

        out.filename = file.filename
        out.lineno = file.lineno
        out.ret_val.filename = out.filename
        out.ret_val.lineno = out.lineno

        # Check if recursive, elemental or pure

        m = re.search(dummy_types_re, cl)
        if m != None:
            out.attributes.append(m.group())
            cl = dummy_types_re.sub('', cl)

        # Get return type, if present
        cl = cl.strip()
        if re.match(types_re, cl) != None:
            out.ret_val.type = re.match(types_re, cl).group()

        # jrk33 - Does function header specify alternate name of
        # return variable?
        ret_var = None
        m = re.search(result_re, cl)
        if m != None:
            ret_var = m.group(1)
            cl = result_re.sub('', cl)

        # Get func name

        cl = funct.sub('', cl)
        out.name = re.search(re.compile(r'\w+'), cl).group()
        log.debug('    module function checking ' + out.name)

        # Default name of return value is function name
        out.ret_val.name = out.name
        # If return type not present, infer type from function name
        if out.ret_val.type == '':
            out.ret_val.type = implicit_type_rule(out.name)

        # Check to see if there are any arguments

        # Find "(" followed by anything else than ")"
        if re.search(r'\([^\)]+', cl) != None:
            has_args = 1
        else:
            has_args = 0

        if has_args:
            # get argument list

            # substitue 'consecutive words' by '' in cl, at most 1 time
            cl = re.sub(r'\w+', '', cl, count=1)
            argl = re.split(r'[\W]+', cl)

            del (argl[0])
            del (argl[len(argl) - 1])

            while cl.strip() == '' or re.search(r'&', cl) != None:
                cl = file.next()
                if cl.startswith('_COMMENT'):
                    cl = file.next()
                if cl.strip() == '':
                    continue
                arglt = re.split(r'[\W]+', cl)
                del (arglt[len(arglt) - 1])
                for a in arglt:
                    argl.append(a.lower())
        else:
            argl = []

        argl = list(map(lambda s: s.lower(), argl))

        # Get next line, and check each possibility in turn

        in_block_doc = False
        had_block_doc = False

        cl = file.next()

        cont = 0
        subroutine_lines = []
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

            # contains statement
            check = check_cont(cl, file)
            if check[0] is not None:
                cont = 1
                cl = check[1]

            if cont == 0:
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

            else:
                while True :
                    # Subroutine definition
                    check = check_subt(cl, file)
                    if check[0] is not None:
                        # Discard contained subroutine
                        cl = check[1]
                        continue

                    # Function definition
                    check = check_funct(cl, file)
                    if check[0] is not None:
                        # Discard contained function
                        cl = check[1]
                        continue
                    break

            m = re.match(funct_end, cl)

            subroutine_lines.append(cl)
            if m == None:
                cl = file.next()
                continue
            else:
                if doc_plugin_module is not None:
                    extra_doc = doc_plugin_module.doc_plugin(subroutine_lines, out.name, 'function')
                    out.doc.extend(extra_doc)
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

        if hold_doc:
            for line in hold_doc[:]:
                match = re.match(doxygen_param_group, line)
                if match and match.group(1) == 'returns':
                    comm = '%s %s'%(match.group(3), match.group(4))
                    out.ret_val.doxygen = comm
                    hold_doc.remove(line)

        implicit_to_explicit_arguments(argl, ag_temp)

        out.arguments = ag_temp
        out.arguments.sort(key=lambda x: argl.index(x.name.lower()))

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


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_type(cl, file):
    #    global hold_doc

    out = Type()
    m = re.match(type_re, cl)
    current_access = None
    cont = 0

    if m is not None:

        out.filename = file.filename
        out.lineno = file.lineno

        # jrk33 - see if it's a global variable of this type.
        # if so, do nothing - it will be found by check_decl
        if decl.match(cl) != None:
            return [None, cl]

        # if hold_doc != None:
        #            for line in hold_doc:
        #                out.doc.append(line)
        #            hold_doc = None

        # Get type name
        cl = type_re.sub('', cl)

        # Check if there are any type attributes
        out.attributes = []
        if m.group(1):
            out.attributes = split_attribs(m.group(1))

        out.name = re.search(re.compile(r'\w+'), cl).group()
        log.info('parser reading type %s' % out.name)

        # Get next line, and check each possibility in turn

        cl = file.next()

        while re.match(type_end, cl) == None:

            # contains statement
            check = check_cont(cl, file)
            if check[0] != None:
                log.debug('parser reading type %s bound procedures', out.name)
                cont = 1
                cl = check[1]
                continue

            if cont == 0:

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

            else:

                check = check_binding(cl, file)
                if check[0] != None:
                    out.bindings.extend(check[0])
                    cl = check[1]
                    continue

            if cl.lower() == 'public':
                current_access = 'public'

            elif cl.lower() == 'private':
                current_access = 'private'

            cl = file.next()

        cl = file.next()

        if any(bind.name == "assignment(=)" for bind in out.bindings):
            out.attributes.append("has_assignment")
        out.lineno = slice(out.lineno, file.lineno - 1)
        return [out, cl]
    else:
        return [None, cl]


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_interface(cl, file):
    global hold_doc

    out = Interface()

    if re.match(iface, cl) == None:
        return [None, cl]

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


def check_interface_decl(cl, file):
    global doc_plugin_module
    out = Interface()

    if cl and re.match(iface, cl) != None:

        out.filename = file.filename
        out.lineno = file.lineno

        cl = file.next()
        while re.match(iface_end, cl) == None:

            # Subroutine declaration
            check = check_subt(cl, file)
            if check[0] != None:
                out.procedures.append(check[0])
                cl = check[1]
                continue

            # Function declaration
            check = check_funct(cl, file)
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


def check_abstract_interface(cl, file):
    global doc_plugin_module
    out = Interface()

    if (not cl) or re.match(abstract_iface, cl) == None:
        return [None, cl]

    out.attributes.append('abstract')
    out.filename = file.filename
    out.lineno = file.lineno

    cl = file.next()
    while re.match(iface_end, cl) == None:

        # Subroutine declaration
        check = check_subt(cl, file)
        if check[0] != None:
            check[0].attributes.append('abstract')
            out.procedures.append(check[0])
            cl = check[1]
            continue

        # Function declaration
        check = check_funct(cl, file)
        if check[0] != None:
            check[0].attributes.append('abstract')
            out.procedures.append(check[0])
            cl = check[1]
            continue

        cl = file.next()

    cl = file.next()

    out.lineno = slice(out.lineno, file.lineno - 1)
    return [out, cl]


def check_prototype(cl, file):
    m = prototype.match(cl)
    if m != None:
        out = map(lambda s: s.strip().lower(), m.group(2).split(','))
        out = [Prototype(name=name, lineno=file.lineno, filename=file.filename) for name in out]

        cl = file.next()
        return [out, cl]

    else:
        return [None, cl]


def check_binding(cl, file):
    m = binding.match(cl)
    if m == None:
        return check_deferred_binding(cl, file)

    type = m.group(1).strip().lower()
    attrs = m.group(4)
    bindings = m.group(6)
    if attrs:
        attrs = [a.strip().lower() for a in attrs.split(',')]
    out = []
    if type == 'generic':
        name, targets = bindings.split('=>')
        name = name.strip().lower()
        log.debug('found generic binding %s => %s', name, targets)
        out.append(Binding(
            name=name,
            lineno=file.lineno,
            filename=file.filename,
            type=type,
            attributes=attrs,
            procedures=[
                Prototype(
                    name=t.strip().lower(),
                    lineno=file.lineno,
                    filename=file.filename
                )
                for t in targets.split(',')
            ],
        ))
    else:
        for b in bindings.split(','):
            name, *target = [ word.strip().lower() for word in b.split('=>')]
            name = name.strip().lower()
            target = target[0] if target else name
            log.debug('found %s binding %s => %s', type, name, target)
            out.append(Binding(
                name=name,
                lineno=file.lineno,
                filename=file.filename,
                type=type,
                attributes=attrs,
                procedures=[
                    Prototype(
                        name=target.strip().lower(),
                        lineno=file.lineno,
                        filename=file.filename,
                    ),
                ],
            ))
    cl = file.next()
    return [out, cl]


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def check_deferred_binding(cl, file):
    m = deferred_binding.match(cl)
    if m == None:
        return [None, cl]

    type = m.group(1).strip().lower()
    interface = m.group(2).strip().lower()
    attrs = m.group(5)
    name = m.group(7)
    if attrs:
        attrs = [a.strip().lower() for a in attrs.split(',')]
    out = []
    log.debug('found deferred %s(%s) binding %s', type, interface, name)
    out.append(Binding(
        name=name,
        lineno=file.lineno,
        filename=file.filename,
        type=type,
        attributes=attrs,
        procedures=[
            Prototype(
                name=interface,
                lineno=file.lineno,
                filename=file.filename
            ),
        ],
    ))
    cl = file.next()
    return [out, cl]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++


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


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
    global hold_doc
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
        nl = [name.strip() for name in nl]

        alist = []
        for j in range(len(atrl)):
            alist.append(atrl[j])

        cl = file.next()
        check = check_doc(cl, file)

        doxygen_map = {}
        if hold_doc:
            for line in hold_doc[:]:
                match = re.match(doxygen_param_group, line)
                if match:
                    if match.group(1) == 'param':
                        direction = match.group(2)
                        name = match.group(3)
                        comm = match.group(4)
                        # Check if all names in name_list are present in nl
                        for single_name in name.split(','):
                            if single_name in nl:
                                try:
                                    hold_doc.remove(line)
                                except ValueError:
                                    pass
                                doxygen_map[single_name] = ' '.join([comm, direction]).strip(' ')

        dc = []

        while check[0] != None:
            # Doc comment
            dc.append(check[0])
            cl = check[1]
            check = check_doc(cl, file)

        cl = check[1]

        for i, arg_name in enumerate(nl):
            try:
                doxy_doc = doxygen_map[arg_name]
            except KeyError:
                doxy_doc = ''

            temp = Argument(name=arg_name, doc=dc, type=tp, attributes=alist[:],
                            filename=filename, lineno=lineno, doxygen=doxy_doc)

            # Append dimension if necessary
            if sizes[i] != '':
                temp.attributes.append('dimension' + sizes[i])
            out.append(temp)

        return [out, cl]
    else:
        return [None, cl]


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++


def read_files(args, doc_plugin_filename=None):
    global doc_plugin_module
    global hold_doc

    if doc_plugin_filename is not None:
        sys.path.insert(0, os.path.dirname(doc_plugin_filename))
        doc_plugin_module = __import__(os.path.splitext(os.path.basename(doc_plugin_filename))[0])
        sys.path = sys.path[1:]
    else:
        doc_plugin_module = None

    root = Root()

    for fn in args:

        fname = fn

        # Open the filename for reading

        log.debug('processing file ' + fname)
        file = F90File(fname)

        # Get first line

        cline = file.next()

        while cline != None:

            # programs
            check = check_program(cline, file)
            if check[0] != None:
                log.debug('  program ' + check[0].name)
                root.programs.append(check[0])
                cline = check[1]
                continue

            # modules/submodules
            check = check_module(cline, root.modules, file)
            if check[0] != None:
                log.debug('  module ' + check[0].name)
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
                # log.debug('  subroutine ' + check[0].name)
                root.procedures.append(check[0])
                cline = check[1]
                continue

            # stand-alone functions
            check = check_funct(cline, file)
            if check[0] != None:
                # log.debug('  function ' + check[0].name)
                root.procedures.append(check[0])
                cline = check[1]
                continue

            cline = file.next()

    # apply some rules to the parsed tree
    root = fix_argument_attributes(root)
    root = LowerCaseConverter().visit(root)
    root = RepeatedInterfaceCollapser().visit(root)
    return root


def dump_package(root, pkg_name, class_names, json_file):
    package = []
    for module in root.modules:
        mod = {}
        mod["name"] = module.name
        mod["package"] = pkg_name
        mod["types"] = []
        for typ in module.types:
            if typ.is_external:
                continue
            t = {"name" : typ.orig_name}
            if typ.orig_name in class_names:
                t["class_name"] = class_names[typ.orig_name]
            t["attributes"] = typ.attributes
            mod["types"].append(t)

        package.append(mod)

    with open(json_file, 'w') as f:
        json.dump(package, f)


def add_external_packages(root, class_names, external_packages):
    for package in external_packages:
        with open(package, 'r') as f:
            modules = json.load(f)

        for mod in modules:
            module = Module()
            module.name = mod["name"]
            module.filename = ""
            module.is_external = True
            for typ in mod["types"]:
                new_type = Type(name=typ["name"], mod_name=module.name)
                new_type.attributes = typ["attributes"]
                new_type.py_mod_name = mod["package"]
                new_type.is_external = True
                module.types.append(new_type)
                if "class_name" in typ:
                    class_names[typ["name"]] = typ["class_name"]
            root.modules.append(module)

    # apply some rules to the parsed tree
    root = fix_argument_attributes(root)
    root = LowerCaseConverter().visit(root)
    root = RepeatedInterfaceCollapser().visit(root)

    return root
