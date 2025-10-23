#=======================================================================
#                   define the compiler names
#=======================================================================
include ../make.inc

PY_MOD      = pywrapper
MOD_PREFIX  = _${PY_MOD}_sign
F90_SRC     = main.f90
OBJ         = $(F90_SRC:.f90=.o)
SIGNATURES	= _signatures
F2PYFLAGS   = --build-dir build
ifeq ($(F90),gfortran)
		LINK		= -lgfortran
endif
.PHONY: all clean

all: test

clean:
	rm -rf *.mod *.smod *.o f90wrap*.f90 ${PY_MOD}_*.py _${PY_MOD}*.so __pycache__/ .f2py_f2cmap build ${PY_MOD}_*/ ${SIGNATURES}*.pyf

%.o: %.f90
	${F90} ${F90FLAGS} -c $< -o $@

${SIGNATURES}: ${F90_SRC}
	@for SUFFIX in f2py_f90wrap_distutils f2py_f90wrap_meson f2py_distutils f2py_meson; do \
		f2py ${F90_SRC} -m ${MOD_PREFIX}_$$SUFFIX -h ${SIGNATURES}_$$SUFFIX.pyf; \
	done

f2py-f90wrap-distutils: ${OBJ} ${SIGNATURES}
	@SUFFIX=f2py_f90wrap_distutils; \
	CFLAGS="${CFLAGS}" f2py-f90wrap -c -m ${MOD_PREFIX}_$$SUFFIX ${F2PYFLAGS} --backend distutils ${LINK} ${OBJ} ${SIGNATURES}_$$SUFFIX.pyf

f2py-f90wrap-meson: ${OBJ} ${SIGNATURES}
	@SUFFIX=f2py_f90wrap_meson; \
	CFLAGS="${CFLAGS}" f2py-f90wrap -c -m ${MOD_PREFIX}_$$SUFFIX ${F2PYFLAGS} --backend meson ${LINK} ${OBJ} ${SIGNATURES}_$$SUFFIX.pyf

f2py-distutils: ${OBJ} ${SIGNATURES}
	@SUFFIX=f2py_distutils; \
	CFLAGS="${CFLAGS}" f2py -c -m ${MOD_PREFIX}_$$SUFFIX ${F2PYFLAGS} --backend distutils ${LINK} ${OBJ} ${SIGNATURES}_$$SUFFIX.pyf

f2py-meson: ${OBJ} ${SIGNATURES}
	@SUFFIX=f2py_meson; \
	CFLAGS="${CFLAGS}" f2py -c -m ${MOD_PREFIX}_$$SUFFIX ${F2PYFLAGS} --backend meson ${LINK} ${OBJ} ${SIGNATURES}_$$SUFFIX.pyf

test-f2py-distutils: f2py-distutils
	${PYTHON} tests_common.py
    # This document what seems to be a limitation in f2py
	${PYTHON} tests_f2py_distutils.py

test-f2py-meson: f2py-meson
    # This document what seems to be a limitation in f2py
	${PYTHON} tests_f2py_meson.py

test-f2py-f90wrap-distutils: f2py-f90wrap-distutils
	${PYTHON} tests_common.py
    # The limitation present in f2py does not appear when using f2py-f90wrap
	${PYTHON} tests_f2py_f90wrap_distutils.py

test-f2py-f90wrap-meson: f2py-f90wrap-meson
	${PYTHON} tests_common.py
    # The limitation present in f2py does not appear when using f2py-f90wrap
	${PYTHON} tests_f2py_f90wrap_meson.py

test:
	# distutils backend is not available with Python >= 3.12
	@if ${PYTHON}  -c "import sys; exit(0 if sys.version_info < (3,12) else 1)"; then \
		make test-f2py-f90wrap-distutils; \
	fi
	make test-f2py-f90wrap-meson
	# distutils backend is not available with Python >= 3.12
	@if ${PYTHON}  -c "import sys; exit(0 if sys.version_info < (3,12) else 1)"; then \
		make test-f2py-distutils; \
	fi
	make test-f2py-meson
