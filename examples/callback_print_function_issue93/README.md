python function callback
==============

This example illustrate how to set up an external callback so that a python function can be called, namely to capture messages for display in a Jupyter notebook (or passed to a log, etc.).

As of 2019-11 there is nothing specific to f90wrap in the content of this folder (f2py should handle it to). Finding how to set up a callback function registration was a day-long pain for yours truly. This example is written in the hope this will ease the pain for others.

## Related resources


* [Call-back arguments in f2py](https://numpy.org/devdocs/f2py/python-usage.html#call-back-arguments)
* Callback registration to e.g. trap C++ exceptions and report through R, Python, etc. [here](https://github.com/csiro-hydroinformatics/moirai/blob/master/src/reference_handle_map_export.cpp)


## Build

This example has been tested with ```gfortran```  on Linux 64bit.

To build and wrap with f90wrap, use the included ```Makefile```:

```sh
make
```

and before rebuilding, clean-up with:

```sh
make clean
```

A simple unittest is included in ```tests.py```.

Author
------

Jean-Michel Perraud: <per202@csiro.au>


