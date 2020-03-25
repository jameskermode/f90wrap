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

LIBSRC_SOURCES = leveltwomod define fwrap

LIBSRC_FILES = $(addsuffix .f90,${LIBSRC_SOURCES})
LIBSRC_OBJECTS = $(addsuffix .o,${LIBSRC_SOURCES})
LIBSRC_FPP_FILES = $(addsuffix .fpp,${LIBSRC_SOURCES})

.SUFFIXES: .f90 .fpp

.PHONY: all clean test

all: test

clean:
	-rm -f ${LIBSRC_OBJECTS} libsrc.a _mockdt*.so _mockdtpkg*.so *.mod f90wrap*.* *.fpp
	-rm -rf mockdtpkg/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/

.f90.o:
	${F90} ${F90FLAGS} -c $< -o $@

.c.o:
	${CC} ${CFLAGS} -c $< -o $@

.f90.fpp:
	${FPP} ${F90FLAGS} $<  -o $@

libsrc.a: ${LIBSRC_OBJECTS}
	${LIBTOOL} $@ $?

_mockdt.so: libsrc.a ${LIBSRC_FPP_FILES}
	f90wrap -m mockdt ${LIBSRC_FPP_FILES} -k kind_map -v 
	f2py-f90wrap --build-dir . -c -m _mockdt -L. -lsrc f90wrap*.f90

_mockdtpkg.so: libsrc.a ${LIBSRC_FPP_FILES}
	f90wrap -m mockdtpkg ${LIBSRC_FPP_FILES} -k kind_map -v -P
	f2py-f90wrap --build-dir . -c -m _mockdtpkg -L. -lsrc f90wrap*.f90

test: _mockdt.so _mockdtpkg.so
	$(PYTHON) test.py

