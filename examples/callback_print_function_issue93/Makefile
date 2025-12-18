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
PYTHON_MODN = CBF
# mapping between Fortran and C types
# Note that in this case this may be superfluous, and at any rate
# 'character' : {'': 'char'}
# was removed from the kind_map as this was leading to:
# f90wrap: RuntimeError('Unknown combination of type "character" and kind "1024" - add to kind map and try again')
# I could not locate useful documentation for what to do.
KIND_MAP = kind_map

#=======================================================================
#
#=======================================================================

#VPATH	=

#=======================================================================
#       List all source files required for the project
#=======================================================================

# names (without suffix), f90 sources
LIBSRC_SOURCES = cback caller

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
LIBSRC_WRAP_SOURCES = cback caller
# LIBSRC_WRAP_SOURCES = caller

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


all: _${PYTHON_MODN}.so _${PYTHON_MODN}_pkg.so test


clean:
	-rm -rf ${LIBSRC_OBJECTS} ${LIBSRC_FPP_FILES} libsrc.a _${PYTHON_MODN}*.so \
	*.mod *.fpp f90wrap*.f90 f90wrap*.o *.o build/ ${PYTHON_MODN}.py ${PYTHON_MODN}_pkg/


.f90.o:
	${F90} ${F90FLAGS} -c $< -o $@


.c.o:
	${CC} ${CFLAGS} -c $< -o $@


.f90.fpp:
	${FPP} ${FPP_F90FLAGS} $<  -o $@


libsrc.a: ${LIBSRC_OBJECTS}
	${LIBTOOL} $@ $?


_${PYTHON_MODN}.so: libsrc.a ${LIBSRC_FPP_FILES}
	f90wrap -m ${PYTHON_MODN} ${LIBSRC_WRAP_FPP_FILES} -k ${KIND_MAP} -v  --callback pyfunc_print
	f2py-f90wrap $(F2PYFLAGS) -c -m _${PYTHON_MODN} -L. -lsrc f90wrap*.f90 ${LIBSRC_WRAP_FILES}


_${PYTHON_MODN}_pkg.so: libsrc.a ${LIBSRC_FPP_FILES}
	rm -rf build/  # Clean meson build directory before second module
	f90wrap -m ${PYTHON_MODN}_pkg ${LIBSRC_WRAP_FPP_FILES} -k ${KIND_MAP} -v -P --callback pyfunc_print
	f2py-f90wrap $(F2PYFLAGS) -c -m _${PYTHON_MODN}_pkg -L. -lsrc f90wrap*.f90 ${LIBSRC_WRAP_FILES}

test: _${PYTHON_MODN}_pkg.so _${PYTHON_MODN}.so
	${PYTHON} -X faulthandler tests.py
