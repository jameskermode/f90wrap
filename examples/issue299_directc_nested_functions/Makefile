# Issue #299: Direct-C generates nested function definitions
# https://github.com/jameskermode/f90wrap/issues/299

NAME=types
WRAPPER=_types

SOURCES=types.f90

# Generate C code with direct-c mode
$(WRAPPER).c: $(SOURCES)
	python -m f90wrap --direct-c -m $(NAME) $(SOURCES)

# Test that C compilation fails (expected to fail until bug is fixed)
test: $(WRAPPER).c
	@echo "Testing C compilation (expected to FAIL until issue #299 is fixed)..."
	@NUMPY_INC=$$(python -c "import numpy; print(numpy.get_include())") && \
	 PYTHON_INC=$$(python -c "import sysconfig; print(sysconfig.get_path('include'))") && \
	 if gcc -c -I"$$NUMPY_INC" -I"$$PYTHON_INC" $(WRAPPER).c 2>/dev/null; then \
	   echo "UNEXPECTED: C compilation succeeded - issue #299 may be fixed!"; \
	   exit 0; \
	 else \
	   echo "EXPECTED: C compilation failed - issue #299 still present"; \
	   exit 1; \
	 fi

clean:
	rm -rf _build_dir meson.build *.c *.py f90wrap_*.f90 *.o *.mod __pycache__ .f2py_f2cmap

.PHONY: test clean
