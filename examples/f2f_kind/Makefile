#=======================================================================
#                   define the compiler names
#=======================================================================

include ../make.inc
PY_MOD      = pywrapper
F90_SRC     = main.f90
OBJ         = $(F90_SRC:.f90=.o)
F90WRAP_SRC = $(addprefix f90wrap_,${F90_SRC})
WRAPFLAGS   = -v --define PREPROCESSOR_DEFINED_INT=4 --define PREPROCESSOR_DEFINED_REAL=4
F2PY        = f2py-f90wrap
F90FLAGS	+= -cpp -DPREPROCESSOR_DEFINED_INT=4 -DPREPROCESSOR_DEFINED_REAL=4
.PHONY: all clean

all: test

clean:
	rm -rf *.mod *.smod *.o *.a f90wrap*.f90 ${PY_MOD}.py _${PY_MOD}*.so __pycache__/ .f2py_f2cmap build ${PY_MOD}/

%.o: %.f90
	${F90} ${F90FLAGS} -c $< -o $@

libsrc.a: ${OBJ}
	ar src $@ $<

${F90WRAP_SRC}: ${F90_SRC}
	${F90WRAP} -m ${PY_MOD} ${WRAPFLAGS} ${F90_SRC}

f2py: ${F90WRAP_SRC} libsrc.a
	CFLAGS="${CFLAGS}" ${F2PY} -c -m _${PY_MOD} ${F2PYFLAGS} f90wrap_*.f90 -L. -lsrc

test: f2py
	python tests.py
