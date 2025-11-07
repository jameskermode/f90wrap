#=======================================================================
#                   define the compiler names
#=======================================================================

include ../make.inc

ifeq ($(notdir $(F90)),gfortran)
    F90FLAGS     += -x f95-cpp-input
else ifeq ($(notdir $(F90)),nvfortran)
	F90FLAGS     += -Mpreprocess
endif

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
PYTHON_MODN = Example
# mapping between Fortran and C types
KIND_MAP = kind_map

#=======================================================================
#
#=======================================================================

#=======================================================================
#       List all source files required for the project
#=======================================================================

# names (without suffix), f90 sources
LIBSRC_SOURCES = DNAD cyldnad

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
LIBSRC_WRAP_SOURCES = cyldnad DNAD
#LIBSRC_WRAP_SOURCES = cyldnad

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


all: _${PYTHON_MODN}.so _${PYTHON_MODN}_pkg.so


clean:
	-rm -f *.o *.so *.a *.mod *.fpp f90wrap*.f90 Example.py *.pyc
	-rm -rf Example_pkg/ src.*
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/


.f90.o:
	${F90} ${F90FLAGS} -c $< -o $@


.c.o:
	${CC} ${CFLAGS} -c $< -o $@


.f90.fpp:
	${FPP} ${FPP_F90FLAGS} $<  -o $@


libsrc.a: ${LIBSRC_OBJECTS}
	${LIBTOOL} $@ $?


_${PYTHON_MODN}.so: libsrc.a ${LIBSRC_FPP_FILES}
	f90wrap -m ${PYTHON_MODN} ${LIBSRC_WRAP_FPP_FILES} -k ${KIND_MAP} -v
	f2py-f90wrap $(F2PYFLAGS) -c -m _${PYTHON_MODN} -L. -lsrc f90wrap*.f90


_${PYTHON_MODN}_pkg.so: libsrc.a ${LIBSRC_FPP_FILES}
	f90wrap -m ${PYTHON_MODN}_pkg ${LIBSRC_WRAP_FPP_FILES} -k ${KIND_MAP} -v -P
	f2py-f90wrap $(F2PYFLAGS) -c -m _${PYTHON_MODN}_pkg -L. -lsrc f90wrap*.f90


test: all
	${PYTHON} tests.py
