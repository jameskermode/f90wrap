"""Test that pointer arrays are correctly handled.

See issue #302: https://github.com/jameskermode/f90wrap/issues/302
Pointer arrays in derived types cannot be fully wrapped,
but the module should still work for other attributes.
"""
import pointer_mod

# Create a container instance
c = pointer_mod.pointer_mod.container_t()

# The size attribute should be accessible and default to 0
# Note: 'size' is renamed to 'size_bn' to avoid conflict with Python builtins
assert c.size_bn == 0, f"Expected size_bn=0, got {c.size_bn}"

print("Done")
