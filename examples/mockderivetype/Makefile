F90 = gfortran
FPP = gfortran -E
F90FLAGS = -x f95-cpp-input

LIBSRC_SOURCES = leveltwomod define fwrap

LIBSRC_FILES = $(addsuffix .f90,${LIBSRC_SOURCES})
LIBSRC_OBJECTS = $(addsuffix .o,${LIBSRC_SOURCES})
LIBSRC_FPP_FILES = $(addsuffix .fpp,${LIBSRC_SOURCES})

.SUFFIXES: .f90 .fpp

.PHONY: all clean test

all: _mockdt.so

clean:
	-rm ${LIBSRC_OBJECTS} libsrc.a mockdt.so *.mod f90wrap*.f90 *.fpp

test: mockdt.so
	python test.py

.f90.o:	
	${F90} ${F90FLAGS} -c $< -o $@

.f90.fpp:
	${FPP} ${F90FLAGS} $<  -o $@

libsrc.a: ${LIBSRC_OBJECTS}
	libtool -static -o $@ $?

_mockdt.so: libsrc.a ${LIBSRC_FPP_FILES}
	f90wrap -m mockdt ${LIBSRC_FPP_FILES} -v --only do_stuff top_level
	f2py -c -m _mockdt -L. -lsrc f90wrap*.f90


