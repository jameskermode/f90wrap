#!/usr/bin/env python

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

# Parts originally based on:
# f90doc - automatic documentation generator for Fortran 90
# Copyright (C) 2004 Ian Rutt

from __future__ import print_function

import sys
import getopt
import logging
from f90wrap.parser import read_files
from f90wrap.latex import write_latex

def usage():
    print(r"""
f90doc: documentation generator for Fortran 90.
Usage: f90doc [-t <title>] [-a <author>] [-i <intro>] [-n] [-s] [-l] [-f] <filenames>
Options:
    -t: specify a title. 
    -a: specify an author.
    -i: specify intro file to be included
    -n: don't use an intro file
    -s: use short documentation format
    -l: write latex output
    
Notes:
    Output is as LaTeX, to standard output.
    A single LaTeX document is generated from all input
    files.

Markup:
    Prepare your document by adding comments after the things
    to which they refer (or on the same line), prefixed with !*FD
    For comments relating to function return values, use !*FDRV

Copyright (C) 2004 Ian Rutt. f90doc comes with ABSOLUTELY NO WARRANTY.
Modifications for Fortran 95 (c) 2006-2008 James Kermode.

This is free software, and you are welcome to redistribute it under
certain conditions. See LICENCE for details.
""")

def main():

    all_args=sys.argv[1:]

    try:
        optlist, args = getopt.getopt(all_args, 'sa:t:hni:lfpo:v', 'help')
    except getopt.GetoptError:
        # print help information and exit:
        print("f90doc: use -h or --help for help")
        sys.exit(2)

    doc_title='Title'
    doc_author='Authors'
    do_short_doc=False
    do_latex=False
    intro='intro'
    do_header = True

    out_file = '-'
    for o,a in optlist:
        if o=='-t':
            doc_title=a
        if o=='-a':
            doc_author=a
        if o=='-s':
            do_short_doc=True
        if o=='-i':
            intro=a
        if o=='-n':
            intro=None
        if o in ('-h','--help'):
            usage()
            sys.exit()
        if o == '-l':
            do_latex = True
        if o == '-p':
            do_header = False
        if o == '-o':
            out_file = a
        if o == '-v':
            logging.root.setLevel(logging.DEBUG)

    if len(args) < 1:
        sys.stderr.write('You need to supply at one least argument!\n')
        sys.exit()

    root = read_files(args)

    if out_file == '-':
        out_stream = sys.stdout
    else:
        out_stream = open(out_file, 'w')

    if do_latex:
        write_latex(root, doc_title, doc_author, do_short_doc,
                    intro, do_header, out_stream)


if __name__ == "__main__":
    sys.exit(main())
