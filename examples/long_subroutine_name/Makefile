#=======================================================================
#                   define the compiler names
#=======================================================================

CC          = gcc
F90         = gfortran
PYTHON      = python
CFLAGS      = -fPIC
F90FLAGS    = -fPIC
PY_MOD      = pywrapper
F90_SRC     = main.f90
OBJ         = $(F90_SRC:.f90=.o)
F90WRAP_SRC = $(addprefix f90wrap_,${F90_SRC})
WRAPFLAGS   = -v --type-check
F2PYFLAGS   = --build-dir build
F90WRAP     = f90wrap
F2PY        = f2py-f90wrap
.PHONY: all clean

all: test

clean:
	rm -rf *.mod *.smod *.o f90wrap*.f90 ${PY_MOD}.py _${PY_MOD}*.so __pycache__/ .f2py_f2cmap build ${PY_MOD}/

main.o: ${F90_SRC}
	${F90} ${F90FLAGS} -c $< -o $@

%.o: %.f90
	${F90} ${F90FLAGS} -c $< -o $@

${F90WRAP_SRC}: ${OBJ}
	${F90WRAP} -m ${PY_MOD} ${WRAPFLAGS} ${F90_SRC}

f2py: ${F90WRAP_SRC}
	CFLAGS="${CFLAGS}" ${F2PY} -c -m _${PY_MOD} ${F2PYFLAGS} f90wrap_*.f90 *.o

test: f2py
	${PYTHON} test.py
