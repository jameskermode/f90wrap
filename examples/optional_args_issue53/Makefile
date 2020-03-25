FPP=gfortran
#FPP=ifort
FFLAGS=-fPIC
PYTHON = python

%.o : %.f90
	${FPP} ${FFLAGS} -c $< -o $@

all: main.o
	f90wrap -m test -P main.f90
	f2py-f90wrap -c -m _test f90wrap_toplevel.f90 main.o

test: all
	$(PYTHON) run.py

clean:
	-rm *.o f90wrap*.f90 *.so *.mod
	-rm -r test/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/

