.. f90wrap documentation master file, created by
   sphinx-quickstart on Thu May  1 21:29:10 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========================================================================
f90wrap: Fortran to Python interface generator with derived type support
========================================================================
`f90wrap` extends the capabilities of `f2py` by providing support for derived
types and other features specific to the Fortran90 standard. 

It constitutes a layer on top of `f2py`, through which modified Fortran source
files are created for input to `f2py`. 

Installation
============
At this stage, you must get the source from https://github.com/jkermode/f90wrap
and then do ``python setup.py install``. 

At a later date we will provide support for ``pip install f90wrap``.


Basic Usage
===========
There are two primary usages of this package. To create documentation from 
Fortran90 source code, use::

	f90doc --help

Otherwise, to use `f90wrap` to generate Fortran90 code suitable for input to 
`f2py`, use::
	
	f90wrap --help

For more advanced usage, see the link below.

Authors
=======

James Kermode: james.kermode@gmail.com

Contributors
============ 
	
Steven Murray: steven.murray@uwa.edu.au

Copyright
=========

Copyright (C)James Kermode 2011. Released under the GNU General Public
License, version 2.

Originally based on f90doc - automatic documentation generator for
Fortran 90. Copyright (C) 2004 Ian Rutt

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

