FC = gfortran
FCFLAGS = -fPIC
PYTHON = python

%.o : %.f90
	${FC} ${FCFLAGS} -c $< -o $@

all: KIMDispersionEquation.o KIMDispersion_Horton.o
	f90wrap -m itest -k .f2py_f2cmap -P KIMDispersionEquation.f90 KIMDispersion_Horton.f90 -v
	f2py-f90wrap --build-dir build -c -m _itest --f2cmap .f2py_f2cmap \
		KIMDispersionEquation.o KIMDispersion_Horton.o \
		f90wrap_KIMDispersionEquation.f90 f90wrap_KIMDispersion_Horton.f90

test: all
	$(PYTHON) run.py

clean:
	rm -f *.o f90wrap*.f90 *.so *.mod
	rm -rf src.*/
	rm -rf itest/
	-rm -rf src.*/ .libs/ __pycache__/ build/


.PHONY: all clean
