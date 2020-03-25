CC = gcc
F90 = gfortran
FPP = gfortran -E
F90FLAGS = -x f95-cpp-input -fPIC
CFLAGS = -fPIC
PYTHON = python

UNAME = $(shell uname)

ifeq (${UNAME}, Darwin)
  LIBTOOL = libtool -static -o
else
  LIBTOOL = ar src
endif

.PHONY: all clean test

all: test

clean:
	-rm -f _type_bn*.so *.mod f90wrap*.f90 *.fpp *.o *.pyc
	-rm -f type_bn.py
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/

_type_bn.so: type_bn.f90
	f90wrap -m type_bn type_bn.f90 -v
	gfortran -c type_bn.f90
	f2py-f90wrap --build-dir . -c -m _type_bn type_bn.o f90wrap*.f90

test: _type_bn.so
	$(PYTHON) test.py

