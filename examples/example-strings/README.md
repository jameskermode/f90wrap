example-strings
===================

When excluding the character/char mapping in ```kind_map```, strings can be
passed between Fortran and Python in a straightforward manner.

To build and wrap with f90wrap, use the included ```Makefile```:

```
make
```

and before rebuilding, clean-up with:

```
make clean
```

Simple unittests are included in ```tests.py```.


Author
------

David Verelst: <david.verelst@gmail.com>
