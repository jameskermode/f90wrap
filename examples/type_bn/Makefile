CC = gcc
F90 = gfortran
FPP = gfortran -E
F90FLAGS = -x f95-cpp-input -fPIC
CFLAGS = -fPIC

UNAME = $(shell uname)

ifeq (${UNAME}, Darwin)
  LIBTOOL = libtool -static -o
else
  LIBTOOL = ar src
endif

.PHONY: all clean test

all: test

clean:
	-rm _type_bn.so *.mod f90wrap*.f90 *.fpp *.o *.pyc
	-rm -r src.*

_type_bn.so: type_bn.f90
	f90wrap -m type_bn type_bn.f90 -v
	gfortran -c type_bn.f90
	f2py-f90wrap --build-dir . -c -m _type_bn type_bn.o f90wrap*.f90

test: _type_bn.so
	python test.py

