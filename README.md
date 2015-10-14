f90wrap: Fortran to Python interface generator with derived type support
========================================================================

[![Build Status](https://travis-ci.org/jameskermode/f90wrap.svg?branch=master)](https://travis-ci.org/jameskermode/f90wrap)

f90wrap is a tool to automatically generate Python extension modules
which interface to Fortran code that makes use of derived types. It
builds on the capabilities of the popular
[f2py](https://sysbio.ioc.ee/projects/f2py2e/) utility by generating a
simpler Fortran 90 interface to the original Fortran code which is then
suitable for wrapping with f2py, together with a higher-level Pythonic
wrapper that makes the existance of an additional layer transparent to
the final user.

Copyright (C) James Kermode 2011-2015. Released under the GNU General
Public License, version 2. Originally based on f90doc - automatic
documentation generator for Fortran 90. Copyright (C) 2004 Ian Rutt.

If you would like to license the source code under different terms,
please contact James Kermode <james.kermode@gmail.com>

Dependencies
------------

 1.  [Python](http://www.python.org) \>= 2.7 (not yet tested with Python 3.x)
 2.  Recent version of [numpy](http://www.numpy.org) which includes `f2py`
 3.  Fortran compiler - tested with `gfortran` 4.6+ and recent `ifort` 12+

Installation
------------

Installation is as follows:

    git clone https://github.com/jameskermode/f90wrap
    cd f90wrap
    python setup.py install [--prefix PREFIX]

Examples and Testing
--------------------

To test the installation, run `make test` from the `examples/`
directory. You may find the code in the various examples useful.

Case studies
------------

f90wrap has been used to wrap the following large-scale scientific
applications:

 - [QUIP](http://libatoms.github.io/QUIP/) - molecular dynamics code
 - [CASTEP](http://www.castep.org) - electronic structure code

Usage
-----

To use `f90wrap` to wrap a set of Fortran 90 source files and produce
wrappers suitable for input to f2py use:

    f90wrap -m MODULE F90_FILES

where `MODULE` is the name of the Python module you want to produce (e.g.
the name of the Fortran code you are wrapping) and `F90_FILES` is a list
of Fortran 90 source files containing the modules, types and subroutines
you would like to expose via Python.

This will produce two types of output: Fortran 90 wrapper files suitable
for input to `f2py` to produce a low-level Python extension module, and a
high-level Python module desinged to be used together with the
f2py-generated module to give a more Pythonic interface.

One Fortran 90 wrapper file is written for each source file, named
`f90wrap_F90_FILE.f90`, plus possibly an extra file named
`f90wrap_toplevel.f90` if there are any subroutines or functions defined
outside of modules in `F90_FILES`.

To use f2py to compile these wrappers into an extension module, use:

    f2py -c -m _MODULE OBJ_FILES f90wrap_*.f90

where `_MODULE` is the name of the low-level extension module.

Optionally, you can replace `f2py` with `f2py-f90wrap`, which is a
slightly modified version of `f2py` includeed in this distribution
that introduces the following features:

1.  Allow the Fortran `present()` intrinsic function to work correctly with
    optional arguments. If an argument to an f2py wrapped function is
    optional and is not given, replace it with `NULL`.
2.  Allow Fortran routines to raise a RuntimeError exception with a
    message by calling an external function `f90wrap_error_abort`().
    This is implemented using a `setjmp()/longjmp()` trap.
3.  Allow Fortran routines to be interrupted with `Ctrl+C` by installing
    a custom interrupt handler before the call into Fortran is made.
    After the Fortran routine returns, the previous interrupt handler
    is restored.

How f90wrap works
-----------------

There are five steps in the process of wrapping a Fortran 90 routine to
allow it to be called from Python.

1.  The Fortran source files are scanned, building up an
    abstract symbol tree (AST) which describes all the modules, types,
    subroutines and functions found.
2.  The AST is transformed to remove nodes which
    should not be wrapped (e.g. private symbols in modules, routines
    with arguments of a derived type not defined in the project, etc.)
3.  The `f90wrap.f90wrapgen.F90WrapperGenerator` class is used to write
    a simplified Fortran 90 prototype for each routine, with derived
    type arguments replaced by integer arrays containing a
    representation of a pointer to the derived type, in the manner
    described in
    (Pletzer2008)[http://link.aip.org/link/?CSENFA/10/86/1].
	This allows opaque references to the
    true Fortran derived type data structures to be passed back and
    forth between Python and Fortran.
4.  f2py is used to combine the F90 wrappers and the original compiled
    functions into a Python extension module (optionally, f2py can be
    replaced by f2py-f90wrap, a slightly modified version which adds
    support for exception handling and interruption during exceution of
    Fortran code).
5.  The `f90wrap.pywrapgen.PythonWrapperGenerator` class is used to
    write a thin object-oriented layer on top of the f2py generated
    wrapper functions which handles conversion between Python object
    instances and Fortran derived-type variables, converting arguments
    back and forth automatically.

Advanced Features
-----------------

Additional command line arguments can be passed to f90wrap to customize
how the wrappers are generated. See the `examples/` directory to see how
some of the options are used:

    -h, --help            show this help message and exit
    -v, --verbose         set verbosity level [default: None]
    -V, --version         show program's version number and exit
    -p PREFIX, --prefix PREFIX
                          Prefix to prepend to arguments and subroutines.
    -c [CALLBACK [CALLBACK ...]], --callback [CALLBACK [CALLBACK ...]]
                          Names of permitted callback routines.
    -C [CONSTRUCTORS [CONSTRUCTORS ...]], --constructors [CONSTRUCTORS [CONSTRUCTORS ...]]
                          Names of constructor routines.
    -D [DESTRUCTORS [DESTRUCTORS ...]], --destructors [DESTRUCTORS [DESTRUCTORS ...]]
                          Names of destructor routines.
    -k KIND_MAP, --kind-map KIND_MAP
                          File containing Python dictionary in f2py_f2cmap
                          format
    -s STRING_LENGTHS, --string-lengths STRING_LENGTHS
                          "File containing Python dictionary mapping string
                          length names to values
    -S DEFAULT_STRING_LENGTH, --default-string-length DEFAULT_STRING_LENGTH
                          Default length of character strings
    -i INIT_LINES, --init-lines INIT_LINES
                          File containing Python dictionary mapping type names
                          to necessary initialisation code
    -I INIT_FILE, --init-file INIT_FILE
                          Python source file containing code to be added to
                          autogenerated __init__.py
    -A ARGUMENT_NAME_MAP, --argument-name-map ARGUMENT_NAME_MAP
                          File containing Python dictionary to rename Fortran
                          arguments
    --short-names SHORT_NAMES
                          File containing Python dictionary mapping full type
                          names to abbreviations
    -m MOD_NAME, --mod-name MOD_NAME
                          Name of output extension module (without .so
                          extension).
    -M, --move-methods    Convert routines with derived type instance as first
                          agument into class methods
    -P, --package         Generate a Python package instead of a single module
    -a ABORT_FUNC, --abort-func ABORT_FUNC
                          Name of Fortran subroutine to invoke if a fatal error
                          occurs
    --only [ONLY [ONLY ...]]
                          Subroutines to include in wrapper
    --skip [SKIP [SKIP ...]]
                          Subroutines to exclude from wrapper         

Author
------

James Kermode: <james.kermode@gmail.com>

Contributors
------------

- Steven Murray [steven-murray](https://github.com/steven-murray)
- Greg Corbett [Gr3gC0rb3tt](https://github.com/Gr3g-C0rb3tt)
- Bob Fischer [citibob](https://github.com/citibob)
- David Verelst [davidovitch](https://github.com/davidovitch)
- James Orr [jamesorr](https://github.com/jamesorr)
- [yvesch](https://github.com/yvesch)


