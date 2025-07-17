include ../make.inc
all: test

%.o : %.f90
	$(F90) ${CFLAGS} -c $< -o $@

test.o: test.f90

test.py: test.o
	f90wrap -m test test.f90
	f2py-f90wrap $(F2PYFLAGS) -c -m _test f90wrap_toplevel.f90 test.o

test: test.py
	$(PYTHON) run.py

clean:
	-rm -f *.o f90wrap*.f90 *.so *.mod *.pyc
	-rm -rf __pycache__
	-rm -f test.py
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build
