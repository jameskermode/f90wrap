example-charstrings
===================

Because of differences between Fortran and C alike strings and characters it
is not straight forward to exchange (long) strings between Fortran routines
and Python. This example shows a workaround for ASCII characters which are
first converted to integers in Python, and back to characters in Fortran.

Simple example with a wrapped subroutines that convert an integer array
representing ASCII characters to a character array and string (character*) in
Fortran.

For inspiration, alternative methods could be based on:

* <https://gcc.gnu.org/onlinedocs/gfortran/Interoperable-Subroutines-and-Functions.html>
* <http://stackoverflow.com/questions/22620643/how-to-bind-cs-char-argument-in-fortran>
* <http://stackoverflow.com/questions/9686532/arrays-of-strings-in-fortran-c-bridges-using-iso-c-binding>
* <http://stackoverflow.com/questions/22293180/passing-numpy-string-format-arrays-to-fortran-using-f2py>

To build and wrap with f90wrap, use the included ```Makefile```:

```
make
```

and before rebuilding, clean-up with:

```
make clean
```

A simple unittests are included in ```tests.py```.


Author
------

David Verelst: <david.Verelst@gmail.com>

