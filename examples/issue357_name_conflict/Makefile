# Issue #357: Name conflict between module variables and function arguments
# https://github.com/jameskermode/f90wrap/issues/357
# FIXED: Wrapper correctly handles shadowing of module variables by function args

include ../make.inc

NAME = base
PYTHON ?= python

UNAME = $(shell uname)
ifeq (${UNAME}, Darwin)
  LIBTOOL = libtool -static -o
else
  LIBTOOL = ar src
endif

.PHONY: all test clean

all: _${NAME}.so

${NAME}.o: ${NAME}.f90
	${F90} ${F90FLAGS} -c $< -o $@

libsrc.a: ${NAME}.o
	${LIBTOOL} $@ $?

_${NAME}.so: libsrc.a
	f90wrap -m ${NAME} ${NAME}.f90 -v
	f2py-f90wrap $(F2PYFLAGS) -c -m _${NAME} -L. -lsrc f90wrap*.f90

test: _${NAME}.so
	$(PYTHON) run.py

clean:
	-rm -f *.o *.mod *.a *.so *.fpp f90wrap*.f90 ${NAME}.py
	-rm -rf ${NAME}_pkg/ src.*/ .f2py_f2cmap .libs/ __pycache__/ build/
