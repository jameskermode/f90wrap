"""Test that pointer arrays are correctly handled.

See issue #302: https://github.com/jameskermode/f90wrap/issues/302
PR #343 added support for pointer arrays in derived types in Direct-C mode.
In standard mode, the module should still work for other attributes.
"""
import pointer_mod

# Create a container instance
c = pointer_mod.pointer_mod.container_t()

# The size attribute should be accessible and default to 0
# Note: 'size' is renamed to 'size_bn' to avoid conflict with Python builtins
assert c.size_bn == 0, f"Expected size_bn=0, got {c.size_bn}"

# Initialize the container with data
c.init(n=5)
assert c.size_bn == 5, f"Expected size_bn=5, got {c.size_bn}"

# Test that the pointer array data is accessible
assert len(c.data) == 5, f"Expected len(data)=5, got {len(c.data)}"
assert all(abs(v) < 1e-10 for v in c.data), f"Expected all zeros, got {c.data}"

# Clean up
c.free()
assert c.size_bn == 0, f"Expected size_bn=0 after free, got {c.size_bn}"

print("Done")
