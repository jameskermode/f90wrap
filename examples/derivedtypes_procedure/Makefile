FC = gfortran
CFLAGS = -fPIC
PYTHON = python

all: test

%.o : %.f90
	${FC} ${CFLAGS} -c $< -o $@

library.o: library.f90

tests.py: library.o
	python -m f90wrap -m library library.f90 -v
	python -m f90wrap --f2py-f90wrap -c -m _library f90wrap_*.f90 library.o

test: tests.py
	$(PYTHON) tests.py

clean:
	-rm -f *.o f90wrap*.f90 *.so *.mod *.pyc
	-rm -rf __pycache__
	-rm -f library.py
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/
