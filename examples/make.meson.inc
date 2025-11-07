CC        ?= gcc
F90       ?= gfortran
F90FLAGS  = -fPIC
FPP       := gfortran -xf95-cpp-input -E
PYTHON    = python
#NAME      = name
WRAPFLAGS := -v
F2PYFLAGS := --lower
DIRECTC   ?= no

ifeq ($(notdir $(F90)),gfortran)
	F90FLAGS  := $(F90FLAGS) -xf95-cpp-input
else ifeq ($(notdir $(F90)),nvfortran)
	F90FLAGS  := $(F90FLAGS) -Mpreprocess
endif


.SUFFIXES: .f90 .fpp .f .F90 .F

.PHONY: all clean

all: test

UNAME = $(shell uname)
ifeq (${UNAME}, Darwin)
   SEDI := gsed -i
else
   SEDI := sed -i
endif
SEDA := "/additional_flags/a add_project_arguments('$$item', language: 'fortran')"

LIBSRC_WRAP_FILES := $(if $(WRAP_SOURCES),$(WRAP_SOURCES),$(filter-out f90wrap_%.f90, $(wildcard *.f90 *.f *.F90)))
define _to_fpp
$(if $(filter %.fpp %.FPP,$1),$1,$(patsubst %.F90,%.fpp,$(patsubst %.f90,%.fpp,$(patsubst %.F,%.fpp,$(patsubst %.f,%.fpp,$1)))))
endef

LIBSRC_WRAP_FPP_FILES := $(foreach src,$(LIBSRC_WRAP_FILES),$(call _to_fpp,$(src)))


.f90.fpp:
	${FPP} $<  -o $@
.F90.fpp:
	${FPP} $<  -o $@
.f.fpp:
	${FPP} $<  -o $@
.F.fpp:
	${FPP} $<  -o $@


# Direct-C mode requires CFLAGS
ifeq ($(DIRECTC),yes)
    PYTHON_INCLUDES := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
    NUMPY_INCLUDES  := $(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")
    CFLAGS          := -fPIC -I$(PYTHON_INCLUDES) -I$(NUMPY_INCLUDES)
    UNAME           := $(shell uname)
    ifeq ($(UNAME),Darwin)
        LDFLAGS     := -bundle -undefined dynamic_lookup -lgfortran
    else
        LDFLAGS     := -shared -lgfortran
    endif
endif

extension: ${LIBSRC_WRAP_FPP_FILES}
ifeq ($(DIRECTC),yes)
	${PYTHON} -m f90wrap --direct-c -m ${NAME} ${LIBSRC_WRAP_FPP_FILES} ${WRAPFLAGS}
else
	${PYTHON} -m f90wrap -m ${NAME} ${LIBSRC_WRAP_FPP_FILES} ${WRAPFLAGS}
	${PYTHON} -m f90wrap --f2py-f90wrap -m _${NAME} f90wrap_*.f90 ${F2PYFLAGS}
endif

build: extension
ifeq ($(DIRECTC),yes)
	@echo 'Building Direct-C extension via meson...'
	@cp ../meson.build.directc.template meson.build
	${SEDI} "s/\$${name}/'${NAME}'/" meson.build
	${SEDI} "s@\$${source}@'${LIBSRC_WRAP_FILES} $(shell ls f90wrap_*.f90)'@" meson.build
	for item in ${F90FLAGS}; do \
		${SEDI} ${SEDA} meson.build; done
	meson setup --wipe _build_dir
	meson compile -C _build_dir
	mv _build_dir/_${NAME}*.so .
else
	@echo 'Building f2py extension via meson...'
	@cp ../meson.build.template meson.build
	${SEDI} "s/\$${name}/'_${NAME}'/" meson.build
	${SEDI} "s@\$${source}@'${LIBSRC_WRAP_FILES} $(shell ls f90wrap_*.f90)'@" meson.build
	for item in ${F90FLAGS}; do \
		${SEDI} ${SEDA} meson.build; done
	meson setup --wipe _build_dir
	meson compile -C _build_dir
	mv _build_dir/_${NAME}*.so .
endif

clean:
	-rm -f f90wrap_*.f90 *.o *.mod *.fpp *.a
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build
	-rm -rf _${NAME}* ${NAME}*/ ${NAME}.py
	-rm -rf _build_dir
	-rm -f meson.build*
