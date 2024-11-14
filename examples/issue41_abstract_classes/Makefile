FC = gfortran
FCFLAGS = -fPIC
PYTHON = python

%.o : %.f90
	${FC} ${FCFLAGS} -c $< -o $@

all: myclass_base.o
	f90wrap -m itest -P myclass_base.f90 -v
	f2py-f90wrap --fcompiler=$(FC) --build-dir . -c -m _itest myclass_base.o

test: all
	$(PYTHON) run.py

clean:
	rm -f *.o f90wrap*.f90 *.so *.mod
	rm -rf src.*/
	rm -rf itest/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/
