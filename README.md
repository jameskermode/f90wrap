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

Copyright (C) James Kermode 2011-2020. Released under the GNU Lesser General
Public License, version 3. Parts originally based on f90doc - automatic
documentation generator for Fortran 90. Copyright (C) 2004 Ian Rutt.

If you would like to license the source code under different terms,
please contact James Kermode <james.kermode@gmail.com>
Dependencies
------------

 1.  [Python](http://www.python.org) 3.6+ (Python 2.7 no longer supported)
 2.  Recent version of [numpy](http://www.numpy.org) which includes `f2py`
 3.  Fortran compiler - tested with `gfortran` 4.6+ and recent `ifort` 12+

Installation
------------

For the latest stable release, install with either `pip`:

```
pip install f90wrap
```

For the development version, installation is as follows:

```
pip install git+https://github.com/jameskermode/f90wrap
```

Note that if your Fortran 90 compiler has a non-standard name
(e.g. gfortran-9) then you need to set the `F90` environment variable
prior to installing f90wrap to ensure it uses the correct one, e.g.

```
F90=gfortran-9 pip install f90wrap
```

Examples and Testing
--------------------

To test the installation, run `make test` from the `examples/`
directory. You may find the code in the various examples useful.

Citing f90wrap
--------------

If you find `f90wrap` useful in academic work, please cite the following
(open access) publication:

> J. R. Kermode, f90wrap: an automated tool for constructing 
> deep Python interfaces to modern Fortran codes. 
> J. Phys. Condens. Matter (2020) 
>[doi:10.1088/1361-648X/ab82d2](https://dx.doi.org/10.1088/1361-648X/ab82d2)

BibTeX entry:

```bibtex

@ARTICLE{Kermode2020-f90wrap,
  title    = "f90wrap: an automated tool for constructing deep Python
              interfaces to modern Fortran codes",
  author   = "Kermode, James R",
  journal  = "J. Phys. Condens. Matter",
  month    =  mar,
  year     =  2020,
  keywords = "Fortran; Interfacing; Interoperability; Python; Wrapping codes;
              f2py",
  language = "en",
  issn     = "0953-8984, 1361-648X",
  pmid     = "32209737",
  doi      = "10.1088/1361-648X/ab82d2"
}

```

Case studies
------------

f90wrap has been used to wrap the following large-scale scientific
applications:

 - [QUIP](http://libatoms.github.io/QUIP/) - molecular dynamics code
 - [CASTEP](http://www.castep.org) - electronic structure code

See this [Jupyter notebook](https://github.com/jameskermode/f90wrap/blob/master/docs/tutorials/f90wrap-demo-feb-2020.ipynb) 
from a recent seminar for more details.

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

    f2py -c -m _MODULE OBJ_FILES f90wrap_*.f90 *.o

where `_MODULE` is the name of the low-level extension module.

Optionally, you can replace `f2py` with `f2py-f90wrap`, which is a
slightly modified version of `f2py` included in this distribution
that introduces the following features:

1.  Allow the Fortran `present()` intrinsic function to work correctly with
    optional arguments. If an argument to an f2py wrapped function is
    optional and is not given, replace it with `NULL`.
2.  Allow Fortran routines to raise a RuntimeError exception with a
    message by calling an external function `f90wrap_abort`().
    This is implemented using a `setjmp()/longjmp()` trap.
3.  Allow Fortran routines to be interrupted with `Ctrl+C` by installing
    a custom interrupt handler before the call into Fortran is made.
    After the Fortran routine returns, the previous interrupt handler
    is restored.

Notes
-----

- Unlike standard `f2py`, `f90wrap` converts all `intent(out)` arrays to
`intent(in, out)`. This was a deliberate design decision to allow allocatable and automatic arrays of unknown output size to be used. It is hard in general to work out what size array needs to be allocated, so relying on the the user to pre-allocate from Python is the safest solution.
- Scalar arguments without `intent` are treated as `intent(in)` by `f2py`. To have `inout` scalars, you need to call `f90wrap` with the `--default-to-inout` flag and declare the python variables as 1-length numpy arrays (`numpy.zeros(1)` for example).
- Pointer arguments are not supported.
- Arrays of derived types are currently not fully supported: a workaround is provided for 1D-fixed-length arrays, i.e. `type(a), dimension(b) :: c`. In this case, the super-type `Type_a_Xb_Array` will be created, and the array of types can be accessed through `c.items`. Note that dimension b can not be `:`, but can be a parameter.


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
    (Pletzer2008)[https://doi.org/10.1109/MCSE.2008.94].
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
                            File containting Python dictionary in f2py_f2cmap
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
      --py-mod-names PY_MOD_NAMES
                            File containing Python dictionary mapping Fortran
                            module names to Python ones
      --class-names CLASS_NAMES
                            File containing Python dictionary mapping Fortran type
                            names to Python classes
      --joint-modules JOINT_MODULES
                            File containing Python dictionary mapping modules
                            defining times to list of additional modules defining
                            methods
      -m MOD_NAME, --mod-name MOD_NAME
                            Name of output extension module (without .so
                            extension).
      -M, --move-methods    Convert routines with derived type instance as first
                            agument into class methods
      --shorten-routine-names
                            Remove type name prefix from routine names, e.g.
                            cell_symmetrise() -> symmetrise()
      -P, --package         Generate a Python package instead of a single module
      -a ABORT_FUNC, --abort-func ABORT_FUNC
                            Name of Fortran subroutine to invoke if a fatal error
                            occurs
      --only [ONLY [ONLY ...]]
                            Subroutines to include in wrapper
      --skip [SKIP [SKIP ...]]
                            Subroutines to exclude modules and subroutines from
                            wrapper
      --skip-types [SKIP_TYPES [SKIP_TYPES ...]]
                            Subroutines to exclude types from wrapper
      --force-public [FORCE_PUBLIC [FORCE_PUBLIC ...]]
                            Names which are forced to be make public
      --default-to-inout    Sets all arguments without intent to intent(inout)
      --conf-file CONF_FILE
                            Use Python configuration script to set options
      --documentation-plugin DOCUMENTATION_PLUGIN
                            Use Python script for expanding the documentation of
                            functions and subroutines. All lines of the given tree
                            object are passed to it with a reference to its
                            documentation
      --py-max-line-length PY_MAX_LINE_LENGTH
                            Maximum length of lines in python files written.
                            Default: 80
      --f90-max-line-length F90_MAX_LINE_LENGTH
                            Maximum length of lines in fortan files written.
                            Default: 120
         

Author
------

James Kermode [jameskermode](https://github.com/jameskermode)

Contributors
------------

- Tamas Stenczel [stenczelt](https://github.com/stenczelt)
- Steven Murray [steven-murray](https://github.com/steven-murray)
- Greg Corbett  [gregcorbett](https://github.com/gregcorbett)
- Bob Fischer [citibob](https://github.com/citibob)
- David Verelst [davidovitch](https://github.com/davidovitch)
- James Orr [jamesorr](https://github.com/jamesorr)
- [yvesch](https://github.com/yvesch)
- [Matthias Cuntz](https://github.com/mcuntz)
- Balthasar Reuter [reuterbal](https://github.com/reuterbal)
