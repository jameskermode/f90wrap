#=======================================================================
#                   define the compiler names
#=======================================================================
include ../make.inc

PY_MOD      = alloc_mod
F90_SRC     = alloc_mod.f90
OBJ         = $(F90_SRC:.f90=.o)
F90WRAP_SRC = $(addprefix f90wrap_,${F90_SRC})
WRAPFLAGS   = -v
F2PY        = f2py-f90wrap
SRC_AR      = libsrc.a
.PHONY: all clean

all: test

clean:
	rm -rf *.mod *.smod *.o ${SRC_AR} f90wrap*.f90 ${PY_MOD}.py _${PY_MOD}*.so __pycache__/ .f2py_f2cmap build ${PY_MOD}/

alloc_mod.o: ${F90_SRC}
	${F90} ${F90FLAGS} -c $< -o $@

%.o: %.f90
	${F90} ${F90FLAGS} -c $< -o $@

${F90WRAP_SRC}: ${OBJ}
	${F90WRAP} -m ${PY_MOD} ${WRAPFLAGS} ${F90_SRC}

${SRC_AR}: ${OBJ}
	ar rcs $@ ${OBJ}

f2py: ${F90WRAP_SRC} ${SRC_AR}
	CFLAGS="${CFLAGS}" ${F2PY} -c -m _${PY_MOD} ${F2PYFLAGS} f90wrap_*.f90 -L. -lsrc

test: f2py
	${PYTHON} tests.py
