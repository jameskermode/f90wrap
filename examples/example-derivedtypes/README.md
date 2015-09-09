example-derived-types
=====================

Example with a set of subroutines, functions, arrays and derived types.
This extends the ```example-array``` example.
It is shown how to access and interact in Python with the Fortran based
derived types and arrays. The example has been tested with both ```ifort```
and ```gfortran```. The compiler can be set in the ```Makefile```.

To build and wrap with f90wrap, use the included ```Makefile```:

```
make
```

and before rebuilding, clean-up with:

```
make clean
```

Simple unittest test cases are included in ```tests.py```.

A sample memory profiling script is included in ```memory_profile.py```.
To profile, run (you will need to have the Python module ```memory_profiler```
installed):

```
python2 -m memory_profiler memory_profile.py
```


Author
------

David Verelst: <david.Verelst@gmail.com>

