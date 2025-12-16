"""Test that pointer arrays are correctly handled.

See issue #302: https://github.com/jameskermode/f90wrap/issues/302
Pointer arrays in derived types cannot be fully wrapped,
but the module should still work for other attributes.
"""
import pointer_mod

# Create a container instance
c = pointer_mod.pointer_mod.Container_T()

# The size attribute should be accessible and default to 0
assert c.size == 0, f"Expected size=0, got {c.size}"

print("Done")
