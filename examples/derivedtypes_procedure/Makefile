include ../make.inc
all: test

.PHONY: build test clean

library.o: library.f90
	${F90} ${F90FLAGS} -c $< -o $@

libsrc.a: library.o
	ar src $@ $<

build: libsrc.a
	python -m f90wrap -m library library.f90 -v
	python -m f90wrap --f2py-f90wrap $(F2PYFLAGS) -c -m _library f90wrap_*.f90 -L. -lsrc

test: build
	$(PYTHON) tests.py

clean:
	-rm -f *.o *.a f90wrap*.f90 *.so *.mod *.pyc
	-rm -rf __pycache__
	-rm -f library.py
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build
