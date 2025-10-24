include ../make.inc

all: test

test: wrapper
	$(PYTHON) test.py

wrapper: wrap-dta_tt wrap-dta_tc wrap-dta_ct wrap-dta_cc

wrap-%: f90wrap_%.f90 %.o
	f2py-f90wrap $(F2PYFLAGS) -c -m _$* --opt="-O0 -g" f90wrap_$*.f90 $*.o

f90wrap_%.f90 %.py: %.o %.f90
	f90wrap -m $* $*.f90 -v

%.o: %.f90
	$(F90) $(FCFLAGS) -c -g -O0 $< -o $@

clean:
	rm -f *.o f90wrap*.f90 *.so *.mod *.x .f2py_f2cmap
	rm -rf src.*/ .libs/ __pycache__/ build/
	rm -rf dta_*.py
