#=======================================================================
#                   define the compiler names
#=======================================================================
include ../make.inc

PY_MOD      = pywrapper
F90_SRC     = main.f90
OBJ         = $(F90_SRC:.f90=.o)
F2PYFLAGS   = --build-dir build
F2PY        = f2py
ifeq ($(notdir $(F90)),gfortran)
		LINK		= -lgfortran
endif
.PHONY: all clean

all: test

clean:
	rm -rf *.mod *.smod *.o f90wrap*.f90 ${PY_MOD}_*.py _${PY_MOD}*.so __pycache__/ .f2py_f2cmap build ${PY_MOD}_*/

%.o: %.f90
	${F90} ${F90FLAGS} -c $< -o $@

f2py: ${OBJ} ${F90_SRC}
	CFLAGS="${CFLAGS}" ${F2PY} -c -m _${PY_MOD}_no_sign ${F2PYFLAGS} ${F90_SRC}

test: f2py
	${PYTHON} tests_no_sign.py
