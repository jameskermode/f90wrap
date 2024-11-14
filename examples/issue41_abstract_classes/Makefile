FC = gfortran
FCFLAGS = -fPIC
PYTHON = python

%.o : %.f90
	${FC} ${FCFLAGS} -c $< -o $@

all: myclass_base.o myclass_impl.o
	f90wrap -m itest -P myclass_base.f90 myclass_impl.f90 -v
	f2py-f90wrap --build-dir . -c -m _itest \
		f90wrap_myclass_impl.f90 myclass_base.o myclass_impl.o

test: all
	$(PYTHON) run.py

clean:
	rm -f *.o f90wrap*.f90 *.so *.mod
	rm -rf src.*/
	rm -rf itest/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/
