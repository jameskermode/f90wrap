CC        = gcc
F90       = gfortran
F90FLAGS  = -fPIC -xf95-cpp-input 
FPP       = gfortran -E
PYTHON    = python
#NAME      = name
WRAPFLAGS := -v
F2PYFLAGS := --lower

.SUFFIXES: .f90 .fpp .f

.PHONY: all clean

all: test

UNAME = $(shell uname)
ifeq (${UNAME}, Darwin)
   SEDI := gsed -i
else
   SEDI := sed -i
endif
SEDA := "/additional_flags/a add_project_arguments('$$item', language: 'fortran')"

LIBSRC_WRAP_FILES := $(filter-out f90wrap_%.f90, $(wildcard *.f90 *.f *.F90))
LIBSRC_WRAP_FPP_FILES := $(LIBSRC_WRAP_FILES:%.f90=%.fpp)


.f90.fpp:
	${FPP} ${F90FLAGS} $<  -o $@
.f.fpp:
	${FPP} ${F90FLAGS} $<  -o $@


extension: ${LIBSRC_WRAP_FPP_FILES}
	${PYTHON} -m f90wrap -m ${NAME} ${LIBSRC_WRAP_FPP_FILES} ${WRAPFLAGS}
	${PYTHON} -m f90wrap --f2py-f90wrap -m _${NAME} f90wrap_*.f90 ${F2PYFLAGS}

build: extension
	@echo 'Using via meson...'
	@cp ../meson.build.template meson.build
	${SEDI} "s/\$${name}/'_${NAME}'/" meson.build
	${SEDI} "s@\$${source}@'${LIBSRC_WRAP_FILES} $(shell ls f90wrap_*.f90)'@" meson.build
	for item in ${F90FLAGS}; do \
		${SEDI} ${SEDA} meson.build; done
	meson setup --wipe _build_dir
	meson compile -C _build_dir
	mv _build_dir/_${NAME}*.so .

clean: 
	-rm -f f90wrap_*.f90 *.o *.mod *.fpp *.a
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/
	-rm -rf _${NAME}* ${NAME}*/ ${NAME}.py
	-rm -rf _build_dir
	-rm -f meson.build*
