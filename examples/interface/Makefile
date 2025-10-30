include ../make.inc
F90FLAGS += -x f95-cpp-input

all: test

example.o: example.f90
	$(F90) $(F90FLAGS) -c $<

mod: example.o
	f90wrap -m example ./example.f90
	f2py-f90wrap $(F2PYFLAGS) -c -m _example f90wrap_example.f90 example.o

package: example.o
	f90wrap -P -m examplepkg ./example.f90
	f2py-f90wrap $(F2PYFLAGS) -c -m _examplepkg f90wrap_example.f90 example.o

test: mod package
	$(PYTHON) test.py
	$(PYTHON) test_pkg.py

clean:
	-rm -rf examplepkg
	-rm -f example.py
	-rm -f f90wrap*.f90 *.o *.pyc *.mod
	-rm -f _example*.so _examplepkg*.so
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/
