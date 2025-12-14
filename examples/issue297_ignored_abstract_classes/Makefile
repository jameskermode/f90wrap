include ../make.inc

%.o : %.f90
	${F90} ${F90FLAGS} -c $< -I. -o $@

# don't wrap myclass_base.f90 here
all: myclass_base.o main.o
	$(F90WRAP) -m itest -P main.f90 -v
	f2py-f90wrap $(F2PYFLAGS) -c -m _itest f90wrap_main.f90 main.o myclass_base.o

test: all
	$(PYTHON) run.py

clean:
	rm -f *.o f90wrap*.f90 *.so *.mod
	rm -rf src.*/
	rm -rf itest/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/
