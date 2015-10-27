all: mymodule.so

mymodule.so : f90wrap_mycode.f90 mycode.o
	f2py-f90wrap -c -m _mymodule mycode.o f90wrap_mycode.f90

f90wrap_mycode.f90 : mycode.F90
	f90wrap -m mymodule mycode.F90

mycode.o : mycode.F90
	gfortran -c -fPIC mycode.F90

clean:
	-rm *.o *.so f90wrap* mymodule.py

test: mymodule.so
	python example_mymodule.py