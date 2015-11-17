all:
	f90wrap -m elmod elemental_module.f90 -v -k kind_map
	f2py-f90wrap --fcompiler=gfortran -I. --build-dir . -c -m _elmod elemental_module.f90 f90wrap*.f90

clean:
	-rm *.o *.mod elmod.py* _elmod.so f90wrap_elemental_module.*

test: all
	python test.py