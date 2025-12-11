# Issue #301: Complex types fail without kind_map
# https://github.com/jameskermode/f90wrap/issues/301
# FIXED: Complex types now work without requiring a kind_map

NAME=complex_mod
WRAPPER=_complex_mod

SOURCES=complex_mod.f90

# Test standard mode (complex should work without kind_map)
test-standard:
	@echo "Testing standard mode with complex types..."
	@rm -f *.c *.py f90wrap_*.f90 .f2py_f2cmap
	@python -m f90wrap -m $(NAME) $(SOURCES) 2>&1 && \
	  echo "PASS: Standard mode wrapper generation succeeded" || \
	  (echo "FAIL: Standard mode wrapper generation failed"; exit 1)

# Test direct-c mode with C compilation
test-directc:
	@echo "Testing direct-c mode with complex types..."
	@rm -f *.c *.py f90wrap_*.f90 .f2py_f2cmap
	@python -m f90wrap --direct-c -m $(NAME) $(SOURCES) 2>&1 && \
	  echo "PASS: Direct-c mode wrapper generation succeeded" || \
	  (echo "FAIL: Direct-c mode wrapper generation failed"; exit 1)
	@NUMPY_INC=$$(python -c "import numpy; print(numpy.get_include())") && \
	 PYTHON_INC=$$(python -c "import sysconfig; print(sysconfig.get_path('include'))") && \
	 gcc -c -I"$$NUMPY_INC" -I"$$PYTHON_INC" $(WRAPPER).c 2>&1 && \
	  echo "PASS: C compilation succeeded" || \
	  (echo "FAIL: C compilation failed"; exit 1)

test: test-standard test-directc
	@echo "All tests passed!"

clean:
	rm -rf *.c *.py f90wrap_*.f90 *.o *.mod __pycache__ .f2py_f2cmap

.PHONY: test test-standard test-directc clean
