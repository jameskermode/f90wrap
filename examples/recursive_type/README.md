recursive_type
==============

Simple example of a recursive derived type definition; that is the
dervied type contains a pointer to the same type. This is common in
linked lists or tree structures.

This example has been tested with ```gfortran```  on Linux 64bit.

To build and wrap with f90wrap, use the included ```Makefile```:

```
make
```

and before rebuilding, clean-up with:

```
make clean
```

A simple unittest is included in ```tests.py```.


Author
------

Gaetan Kenway: <gaetank@gmail.com>
