# Issue #299: Direct-C generates nested function definitions
# https://github.com/jameskermode/f90wrap/issues/299
# FIXED: This test now verifies the fix works correctly

NAME=types
WRAPPER=_types

SOURCES=types.f90

# Generate C code with direct-c mode
$(WRAPPER).c: $(SOURCES)
	python -m f90wrap --direct-c -m $(NAME) $(SOURCES)

# Test that C compilation succeeds (issue #299 is now fixed)
# Note: Run python from parent dir to avoid types.py shadowing stdlib types module
test: $(WRAPPER).c
	@echo "Testing C compilation for issue #299 fix..."
	@NUMPY_INC=$$(cd .. && python -c "import numpy; print(numpy.get_include())") && \
	 PYTHON_INC=$$(cd .. && python -c "import sysconfig; print(sysconfig.get_path('include'))") && \
	 if gcc -c -I"$$NUMPY_INC" -I"$$PYTHON_INC" $(WRAPPER).c 2>/dev/null; then \
	   echo "PASS: C compilation succeeded - issue #299 is fixed"; \
	   exit 0; \
	 else \
	   echo "FAIL: C compilation failed - regression detected!"; \
	   exit 1; \
	 fi

clean:
	rm -rf _build_dir meson.build *.c *.py f90wrap_*.f90 *.o *.mod __pycache__ .f2py_f2cmap

.PHONY: test clean
