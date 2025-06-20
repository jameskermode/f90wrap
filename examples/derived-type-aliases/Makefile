#=======================================================================
#                   define the compiler names
#=======================================================================

include ../make.inc

UNAME = $(shell uname)

ifeq (${UNAME}, Darwin)
  LIBTOOL = libtool -static -o
else
  LIBTOOL = ar src
endif

# ======================================================================
# PROJECT CONFIG, do not put spaced behind the variables
# ======================================================================
# Python module name
PYTHON_MODN = derived_types_mod
# mapping between Fortran and C types
KIND_MAP = kind_map

#=======================================================================
#
#=======================================================================

#VPATH	=

#=======================================================================
#       List all source files required for the project
#=======================================================================

# names (without suffix), f90 sources
LIBSRC_SOURCES = mytype_mod othertype_mod

# file names
LIBSRC_FILES = $(addsuffix .f90,${LIBSRC_SOURCES})

# object files
LIBSRC_OBJECTS = $(addsuffix .o,${LIBSRC_SOURCES})

# only used when cleaning up
LIBSRC_FPP_FILES = $(addsuffix .fpp,${LIBSRC_SOURCES})

#=======================================================================
#       List all source files that require a Python interface
#=======================================================================

# names (without suffix), f90 sources
LIBSRC_WRAP_SOURCES = mytype_mod othertype_mod

# file names
LIBSRC_WRAP_FILES = $(addsuffix .f90,${LIBSRC_WRAP_SOURCES})

# object files
LIBSRC_WRAP_OBJECTS = $(addsuffix .o,${LIBSRC_WRAP_SOURCES})

# fpp files
LIBSRC_WRAP_FPP_FILES = $(addsuffix .fpp,${LIBSRC_WRAP_SOURCES})

#=======================================================================
#                 Relevant suffixes
#=======================================================================

.SUFFIXES: .f90 .fpp

#=======================================================================
#
#=======================================================================

.PHONY: all clean


all: _mytype.so _othertype.so test


clean:
	-rm  *.o *.a *.so *.mod *.fpp f90wrap*.f90 f90wrap*.o


.f90.o:
	${F90} ${F90FLAGS} -c $< -o $@


.c.o:
	${CC} ${CFLAGS} -c $< -o $@


.f90.fpp:
	${FPP} ${FPP_F90FLAGS} $<  -o $@


libmytype_mod.a: mytype_mod.o
	${LIBTOOL} $@ $?

libothertype_mod.a: othertype_mod.o
	${LIBTOOL} $@ $?


_mytype.so: libmytype_mod.a mytype_mod.f90
	f90wrap -m mytype mytype_mod.f90 -k ${KIND_MAP} -v
	f2py-f90wrap $(F2PYFLAGS) -c -m _mytype -L. -lmytype_mod f90wrap_mytype_mod.f90

_othertype.so: libothertype_mod.a othertype_mod.f90
	f90wrap -m othertype othertype_mod.f90 -k ${KIND_MAP} -v
	f2py-f90wrap $(F2PYFLAGS) -c -m _othertype -L. -lothertype_mod f90wrap_othertype_mod.f90


test: _mytype.so _othertype.so
	${PYTHON} tests.py
