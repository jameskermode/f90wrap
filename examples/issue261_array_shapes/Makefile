FC = gfortran
FCFLAGS = -fPIC
PYTHON = python

all: test

test: wrapper
	$(PYTHON) test.py

wrapper: wrap-array_shapes

wrap-%: f90wrap_%.f90 %.o
	f2py-f90wrap --build-dir . -c -m _$* --opt="-O0 -g" f90wrap_$*.f90 $*.o 

.PRECIOUS: f90wrap_%.f90

f90wrap_%.f90 %.py: %.o %.f90
	f90wrap -m $* $*.f90 -v

%.o: %.f90
	$(FC) $(FCFLAGS) -c -g -O0 $< -o $@

clean:
	rm -f *.o f90wrap*.f90 *.so *.mod *.x .f2py_f2cmap
	rm -rf src.*/ .libs/ __pycache__/
	rm -rf array_shapes.py
