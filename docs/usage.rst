.. _usage:

Usage
=====

Quick overview
--------------

To use `f90wrap` to wrap a set of Fortran 90 source files and produce
wrappers suitable for input to `f2py` use::

	f90wrap -m MODULE F90_FILES

where `MODULE` is the name of the Python module you want to produce
(e.g. the name of the Fortran code you are wrapping) and `F90_FILES`
is a list of Fortran 90 source files containing the modules, types and
subroutines you would like to expose via Python.

This will produce two types of output: Fortran 90 wrapper files
suitable for input to `f2py` to produce a low-level Python extension
module, and a high-level Python module desinged to be used together
with the `f2py`-generated module to give a more Pythonic interface.

One Fortran 90 wrapper file is written for each source file, named
:file:`f90wrap_F90_FILE.f90`, plus possibly an extra file named
:file:`f90wrap_toplevel.f90` if there are any subroutines or functions
defined outside of modules in `F90_FILES`.

To use `f2py` to compile these wrappers into an extension module,
use::

	f2py -c -m _MODULE OBJ_FILES f90wrap_*.f90

where `_MODULE` is the name of the low-level extension module.


How f90wrap works
-----------------

There are five steps in the process of wrapping a Fortran 90
routine to allow it to be called from Python.

1. The Fortran source files are scanned, building up an AST (:mod:`ast
   <abstract symbol tree>`) which describes all the modules, types,
   subroutines and functions found.

2. The AST is :mod:`~f90wrap.transform <transformed>` to remove nodes
   which should not be wrapped (e.g. private symbols in modules,
   routines with arguments of a derived type not defined in the
   project, etc.)

3. The :class:`~f90wrap.f90wrapgen.F90WrapperGenerator` class is used to
   write a simplified Fortran 90 prototype for each routine, with
   derived type arguments replaced by integer arrays containing a
   representation of a pointer to the derived type, in the manner
   described in [Pletzer2008]_. This allows opaque references to the
   true Fortran derived type data structures to be passed back and
   forth between Python and Fortran.

4. `f2py` is used to combine the F90 wrappers and the original
   compiled functions into a Python extension module (optionally,
   `f2py` can be replaced by :ref:`f2py-f90wrap`, a slightly modified
   version which adds support for exception handling and interruption
   during exceution of Fortran code).

5. The :class:`~f90wrap.pywrapgen.PythonWrapperGenerator` class is
   used to write a thin object-oriented layer on
   top of the `f2py` generated wrapper functions which handles
   conversion between Python object instances and Fortran derived-type
   variables, converting arguments back and forth automatically.


Transformation of the symbol tree
---------------------------------

Generation of Fortran 90 wrappers
---------------------------------

All routines which accept derived types arguments are wrapped by
equivalent routines which instead accept integer arrays as opaque
handles.  The Fortran `transfer()` intrinsic is used to convert these
handles into pointers to derived types, as described in
[Pletzer2008]_. The size of the integer array required to store a
pointer is determined automatically at compile-time by the
:func:`f90wrap.sizeoffortran.sizeof_fortran_t` function. In this way
we can access the underlying Fortran structures from Python.

Extra routines are generated to access the values of attributes within
Fortran modules and derived types. For scalars a pair of get and set routines
is created, whilst for arrays a single routine which returns the
shape, memory location and type of the array is output. Derived type
elements within modules or other derived types are also supported, so
that entire hiereachies of types can be wrapped.


Fortran Types and Kinds (-k/--kind-map)
***************************************

Constructors and Destructors (-C/-D)
************************************

Constructor and desctructor routines are handled specially: on
initialisation, a derived type pointer is allocated before the wrapped
routine is invoked, and an opaque reference to this new derived type
is returned. On finalisation the underlying derived type pointer is
deallocated after the wrapped routine returns.


.. _f2py-f90wrap:

f2py-f90wrap
------------

Optionally, you can replace `f2py` with `f2py-f90wrap`, which is
slightly modified version of f2py which introduces the following
features:

  1. Allow the Fortran :c:func:`present` function to work correctly
     with optional arguments.  If an argument to an f2py wrapped
     function is optional and is not given, replace it with ``NULL``.

  2. Allow Fortran routines to raise a :exc:`RuntimeError` exception
     with a message by calling an external function
     :c:func:`f90wrap_error_abort`. This is implemented using a
     :c:func:`setjmp`/ :c:func:`longjmp` trap.

  3. Allow Fortran routines to be interrupted with :kbd:`Ctrl+C` by
     installing a custom interrupt handler before the call into
     Fortran is made. After the Fortran routine returns, the previous
     interrupt handler is restored.


.. [Pletzer2008] Pletzer, A et al., Exposing Fortran Derived Types to C and Other Languages,
   *Computing in Science and Engineering*, **10**, 86 (2008).
   http://link.aip.org/link/?CSENFA/10/86/1
