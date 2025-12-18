include ../make.inc

test :  clean functions.o wrap python
	$(PYTHON) tests.py
functions.o : functions.f90
	$(F90) $(F90FLAGS) -c -O3 functions.f90
wrap :
	f90wrap --default-to-inout -m test_python functions.f90
python :
	f2py-f90wrap $(F2PYFLAGS) -c -m _test_python f90wrap_functions.f90 *.o
clean :
	rm -f *.o *.mod *.so f90wrap* test_python.py
	rm -rf build/
