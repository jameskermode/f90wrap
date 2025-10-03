#=======================================================================
#                   define the compiler names
#=======================================================================

CC       = gcc
F90      = gfortran
#F90      = ifort
PYTHON = python


moddir   = modules
wrap_dir = f90wrap

#=======================================================================
#                     additional flags
#=======================================================================

ifeq ($(F90),gfortran)
	F90FLAGS = #-x f95-cpp-input ==> preprocess NOT NEEDED WITH .F90 EXTENSION
      FCOMP    = gfortran
endif

ifeq ($(F90),ifort)
	F90FLAGS = -fpscomp logicals -module $(moddir)   # use 1 and 0 for True and False
      FCOMP    = intelem             # for f2py
endif

FPP      = gfortran -E
SOFLAGS  = -fPIC       #     ==> universal for ifort, gfortran, pgi

#=======================================================================
#=======================================================================

UNAME = $(shell uname)

ifeq (${UNAME}, Darwin)
  LIBTOOL = libtool -static -o
else
  LIBTOOL = ar src
endif

#=======================================================================
#                 List all source files
#=======================================================================
#     Note: right now, all source files for creating the library (.a)
#           AND the python interface generator are the same. In a realistic
#           situation, we would give python access to only SELECT
#           source files (e.g. top level subroutines) and leave the rest
#           of the fortran code as "hidden" from python, since low-level
#           routines should generally not be accessed once verified
#=======================================================================

#names (without suffix)
LIBSRC_SOURCES   =  Source/BasicDefs/aa0_typelist \
                    Source/BasicDefs/aa1_modules Source/BasicDefs/assign_constants \
                    Source/BasicDefs/aa2_defineAllProperties \
                    Source/HeliSrc/set_defaults

#file names
LIBSRC_FILES     = $(addsuffix .F90,${LIBSRC_SOURCES})

#object files
LIBSRC_OBJECTS   = $(addsuffix .o,${LIBSRC_SOURCES})

#same files, except with fpp extension
LIBSRC_FPP_FILES = $(addsuffix .fpp,${LIBSRC_SOURCES})

#fpp files : address
SRC2_FILES       = $(addprefix '../',$(LIBSRC_FPP_FILES))

kmap_file        = ../kind_map

#=======================================================================
#                 Relevant suffixes
#=======================================================================

.SUFFIXES: .F90 .fpp

#=======================================================================
#                 Makefile default
#=======================================================================

.PHONY: all clean test

all: test
# test run

clean:
	rm -rf $(moddir)/*
	rm -rf $(wrap_dir)/*
	rm -rf mockdt* *.so *.mod
	rm -rf mockdtpkg
	rm -f ${LIBSRC_OBJECTS} libsrc.a $(LIBSRC_FPP_FILES)
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/
	-rm -rf f90wrap/ modules/

.F90.o:
	${F90} ${F90FLAGS} ${SOFLAGS} -c $< -o $@

.c.o:
	${CC} ${SOFLAGS} -c $< -o $@

.F90.fpp:
	${FPP} ${SOFLAGS} $<  -o $@

libsrc.a: ${LIBSRC_OBJECTS}
	${LIBTOOL} $@ $?

#=======================================================================
#shared object (direct compilation, single .py module)
#=======================================================================

_mockdt.so: libsrc.a ${LIBSRC_FPP_FILES}
	mkdir -p f90wrap
	mkdir -p modules
	cd $(wrap_dir) ; f90wrap -m mockdt ${SRC2_FILES} -k $(kmap_file) -v
	cp $(wrap_dir)/*.py .
	f2py-f90wrap --fcompiler=$(FCOMP) -I$(moddir) --build-dir . -c -m _mockdt -L. -lsrc $(wrap_dir)/f90wrap*.f90

#=======================================================================
#a "package", denoted by the -P option (subdirectory with multiple files)
#=======================================================================

_mockdtpkg.so: libsrc.a ${LIBSRC_FPP_FILES}
	cd $(wrap_dir) ; f90wrap -m mockdtpkg ${SRC2_FILES} -k $(kmap_file) -v -P
	f2py-f90wrap --fcompiler=$(FCOMP) -I$(moddir) --build-dir . -c -m _mockdtpkg -L. -lsrc $(wrap_dir)/f90wrap*.f90
	rm -rf mockdtpkg
	mv $(wrap_dir)/mockdtpkg .
	#mv _mockdtpkg.so mockdtpkg/

#=======================================================================
#           All shared objects
#=======================================================================

all_so: _mockdt.so _mockdtpkg.so

#=======================================================================
#           final compilation and running of python script
#=======================================================================

test: all_so
	$(PYTHON) test_module.py
	$(PYTHON) test_package.py
