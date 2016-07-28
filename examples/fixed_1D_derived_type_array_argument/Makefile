test :  clean functions.o wrap python
	python tests.py
functions.o : functions.f90
	gfortran -c -O3 -fPIC functions.f90
wrap :
	f90wrap --default-to-inout -m test_python functions.f90 
python :
	f2py-f90wrap -c -m _test_python f90wrap_functions.f90 *.o
clean :
	rm -f *.o *.mod *.so f90wrap* test_python.py


