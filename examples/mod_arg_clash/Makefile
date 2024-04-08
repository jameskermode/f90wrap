FPP=gfortran
#FPP=ifort
FFLAGS=-fPIC
PYTHON = python

%.o : %.f90
	${FPP} ${FFLAGS} -c $< -o $@

all: test.o
	f90wrap -m test -P test.f90
	f2py-f90wrap -c -m _test f90wrap_test.f90 test.o

test: all
	$(PYTHON) tests.py

clean:
	-rm *.o f90wrap*.f90 *.so *.mod
	-rm -r test/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/

