include ../make.inc

all: mymodule.so

mymodule.so : f90wrap_mycode.f90 mycode.o
	f2py-f90wrap $(F2PYFLAGS) -c -m _mymodule mycode.o f90wrap_mycode.f90

f90wrap_mycode.f90 : mycode.F90
	f90wrap -m mymodule mycode.F90

mycode.o : mycode.F90
	$(F90) $(F90FLAGS) -c mycode.F90

clean:
	-rm -f *.o *.so f90wrap* mymodule.py
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/

test: mymodule.so
	$(PYTHON) example_mymodule.py
