FPP=gfortran
#FPP=ifort
FFLAGS=-fPIC
PYTHON = python

%.o : %.f90
	${FPP} ${FFLAGS} -c $< -o $@

all: testextends.o
	f90wrap -m testextends testextends.f90
	f2py-f90wrap -c -m _testextends f90wrap_*.f90

test: all
	$(PYTHON) testextends.py

clean:
	-rm -f *.o f90wrap*.f90 *.so *.mod testextends.py
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/

