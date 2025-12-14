# Issue #302: Pointer arrays silently skipped in direct-c mode
# https://github.com/jameskermode/f90wrap/issues/302
# FIXED: f90wrap now warns when skipping unsupported features like pointer arrays

NAME=pointer_mod
WRAPPER=_pointer_mod

SOURCES=pointer_mod.f90

# Test that direct-c mode produces a warning for pointer arrays in derived types
test:
	@echo "Testing that f90wrap warns about pointer arrays..."
	@rm -f *.c *.py f90wrap_*.f90 .f2py_f2cmap
	@OUTPUT=$$(python -m f90wrap --direct-c -m $(NAME) $(SOURCES) 2>&1); \
	 echo "$$OUTPUT"; \
	 if echo "$$OUTPUT" | grep -q "skipping container_t.data"; then \
	   echo "PASS: Warning message present for pointer arrays"; \
	 else \
	   echo "FAIL: No warning about pointer arrays - regression!"; \
	   exit 1; \
	 fi

clean:
	rm -rf *.c *.py f90wrap_*.f90 *.o *.mod __pycache__ .f2py_f2cmap

.PHONY: test clean
