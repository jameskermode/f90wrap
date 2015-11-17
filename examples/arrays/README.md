example-arrays
==============

Simple example with a wrapped subroutine that contains input and output arrays.
It simply shows how to access and  interact with an array that is being
initiated in Python, and manipulated by a Fortran subroutine.

This example has been tested with ```gfortran``` and ```ifort``` on Linux 64bit.

To build and wrap with f90wrap, use the included ```Makefile```:

```
make
```

and before rebuilding, clean-up with:

```
make clean
```

A simple unittest is included in ```tests.py```.

A sample memory profiling script is included in ```memory_profile.py```.
To profile, run:

```
python2 -m memory_profiler memory_profile.py
```


Author
------

David Verelst: <david.Verelst@gmail.com>

