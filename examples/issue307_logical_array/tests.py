"""
Test for issue #307: Logical arrays require int32 workaround

This test verifies that:
1. The docstring correctly indicates int32 array (not bool array)
2. int32 arrays work correctly with Fortran logical arrays
"""
import numpy as np
from logical_mod import Logical_Mod as logical_mod

def test_docstring():
    """Check that docstring mentions int32, not bool for logical arrays."""
    doc = logical_mod.get_flags.__doc__
    print("get_flags docstring:")
    print(doc)
    print()

    # The docstring should mention int32, not bool for the logical array
    # This is the fix for issue #307
    assert "int32 array" in doc or "int array" in doc, \
        f"Expected 'int32 array' in docstring, got: {doc}"
    assert "bool array" not in doc, \
        f"Docstring should not say 'bool array': {doc}"

def test_logical_array_works():
    """Test that logical arrays work correctly with int32."""
    n = 10

    # Create int32 output array (not bool!)
    flags = np.zeros(n, dtype=np.int32)

    # Call the Fortran subroutine
    logical_mod.get_flags(n, flags)

    # Verify results: even indices (1-based: 2,4,6,8,10) should be True
    expected = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    np.testing.assert_array_equal(flags, expected)

    # Convert to bool for Python usage
    bool_flags = flags.astype(bool)
    print(f"flags (int32): {flags}")
    print(f"flags (bool):  {bool_flags}")

def test_set_flags():
    """Test input logical array with int32."""
    n = 5

    # Create int32 input array
    flags = np.array([1, 0, 1, 0, 1], dtype=np.int32)

    # Call the Fortran subroutine
    result = logical_mod.set_flags(n, flags)

    # Should count 3 True values
    assert result == 3, f"Expected 3, got {result}"
    print(f"set_flags result: {result}")

if __name__ == "__main__":
    print("Testing issue #307: Logical arrays")
    print("=" * 50)

    test_docstring()
    print("PASS: Docstring correctly shows int32 array")
    print()

    test_logical_array_works()
    print("PASS: get_flags works with int32 arrays")
    print()

    test_set_flags()
    print("PASS: set_flags works with int32 arrays")
    print()

    print("=" * 50)
    print("All tests passed!")
