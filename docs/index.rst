.. f90wrap documentation master file, created by
   sphinx-quickstart on Thu May  1 21:29:10 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========================================================================
f90wrap: Fortran to Python interface generator with derived type support
========================================================================
`f90wrap` extends the capabilities of `f2py` by providing support for derived
types and other features specific to the Fortran90 standard. 

It consists of a layer on top of `f2py`, through which modified
Fortran source files are created for input to `f2py`.

Installation
============

At this stage, you must get the source from https://github.com/jameskermode/f90wrap
and then do ``python setup.py install``. 

At a later date we will provide support for ``pip install f90wrap``.

Features
========

`f90wrap` allows a superset of the Fortran 90/95 language elements
which can be wrapped by `f2py` to be accessed from Python, adding
support for:

  - Subroutines and functions with derived type arguments
  - Access to elements within derived types
  - Access to module data - scalars, arrays and derived types

Case studies
============

`f90wrap` has been used to wrap the following real-world large-scale scientific applications

- `QUIP <http://www.libatoms.org>`_ - molecular dynamics code
- `CASTEP <http://www.castep.org>`_electronic structure code

Basic Usage
===========

To use `f90wrap` to wrap a set of Fortran 90 source files and produce
wrappers suitable for input to `f2py` use::
	
	f90wrap -m MODULE F90_FILES

where `MODULE` is the name of the Python module you want to produce
(e.g. the name of the Fortran code you are wrapping) and `F90_FILES`
is a list of Fortran 90 source files containing the modules, types and
subroutines you would like to expose via Python.

To use `f2py` to compile these wrappers into an extension module,
use::

	f2py -c -m _MODULE OBJ_FILES f90wrap_*.f90

For more advanced usage, see the :ref:`usage` section or the
:file:`examples/` directory in the source distribution.


Authors
=======

James Kermode: james.kermode@gmail.com

Contributors
============ 
	
Steven Murray: steven.murray@uwa.edu.au

Copyright
=========

Copyright (C)James Kermode 2011-2014. Released under the GNU General Public
License, version 2.

Parser originally based on f90doc - automatic documentation generator
for Fortran 90. Copyright (C) 2004 Ian Rutt

These portions of the source code are released under the GNU General
Public License, version 2, http://www.gnu.org/copyleft/gpl.html

If you would like to license the source code under different terms,
please contact James Kermode, james.kermode@gmail.com

When using this software, please cite the following reference:

http://www.jrkermode.co.uk/f90wrap

API Documentation
=================

.. toctree::
   :maxdepth: 2

   apidocs/f90wrap

Notes on Usage
==============
.. toctree::
   :maxdepth: 2
	
   usage

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

