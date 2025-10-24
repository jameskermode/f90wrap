include ../make.inc

%.o : %.f90
	$(F90) ${FCFLAGS} -c $< -o $@

all: alloc_output.o
	${F90WRAP} -m itest -P alloc_output.f90 -v
	f2py-f90wrap ${F2PYFLAGS} -c -m _itest f90wrap_alloc_output.f90 alloc_output.o

test: all
	$(PYTHON) run.py

clean:
	rm -f *.o f90wrap*.f90 *.so *.mod
	rm -rf src.*/
	rm -rf itest/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/
