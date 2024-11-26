FC = gfortran
FCFLAGS = -fPIC
PYTHON = python

all: wrapper

test: wrapper
	$(PYTHON) run.py

wrapper: myclass.o myclass_factory.o
	f90wrap -m itest -P myclass.f90 myclass_factory.f90 -v
	f2py-f90wrap --build-dir . -c -m _itest --opt="-O0 -g" \
		f90wrap_myclass.f90 f90wrap_myclass_factory.f90 \
		myclass.o myclass_factory.o

%.o : %.f90
	$(FC) $(FCFLAGS) -c -g -O0 $< -o $@

clean:
	rm -f *.o f90wrap*.f90 *.so *.mod
	rm -rf src.*/
	rm -rf itest/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/
