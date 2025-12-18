"""
Test for issue #306: Module-level allocatable arrays fail after reallocation

This test verifies that module-level allocatable arrays can be:
1. Allocated and accessed from Python
2. Reallocated with different dimensions and still accessed
"""
import numpy as np
# Import the module instance (lowercase), not the class
from alloc_mod import alloc_mod

def test_allocate_and_access():
    """Test basic allocation and access."""
    print("Test 1: Basic allocation and access")

    # Allocate 3x4 array
    alloc_mod.allocate_array(3, 4)
    alloc_mod.fill_array(1.5)

    # Access the array
    arr = alloc_mod.data_array
    print(f"  Array shape: {arr.shape}")
    print(f"  Array values: {arr[0, 0]}")

    assert arr.shape == (3, 4), f"Expected (3, 4), got {arr.shape}"
    assert np.allclose(arr, 1.5), f"Expected all 1.5, got {arr}"
    print("  PASS")

def test_reallocate_different_size():
    """Test reallocation with different dimensions - this is the issue."""
    print("\nTest 2: Reallocation with different dimensions")

    # First allocation: 2x2
    alloc_mod.allocate_array(2, 2)
    alloc_mod.fill_array(1.0)
    arr1 = alloc_mod.data_array
    print(f"  Initial shape: {arr1.shape}")
    assert arr1.shape == (2, 2), f"Expected (2, 2), got {arr1.shape}"

    # Reallocate to 4x4 (different size!)
    alloc_mod.reallocate_array(4, 4)
    alloc_mod.fill_array(2.0)

    # This is where issue #306 occurs - accessing the array after reallocation
    try:
        arr2 = alloc_mod.data_array
        print(f"  After reallocation shape: {arr2.shape}")
        assert arr2.shape == (4, 4), f"Expected (4, 4), got {arr2.shape}"
        assert np.allclose(arr2, 2.0), f"Expected all 2.0, got {arr2}"
        print("  PASS")
    except ValueError as e:
        print(f"  FAIL: {e}")
        raise

def test_multiple_reallocations():
    """Test multiple reallocations."""
    print("\nTest 3: Multiple reallocations")

    sizes = [(2, 3), (5, 5), (1, 10), (3, 3)]

    for i, (n, m) in enumerate(sizes):
        alloc_mod.reallocate_array(n, m)
        alloc_mod.fill_array(float(i))
        arr = alloc_mod.data_array
        print(f"  Iteration {i}: shape = {arr.shape}")
        assert arr.shape == (n, m), f"Expected ({n}, {m}), got {arr.shape}"

    print("  PASS")

if __name__ == "__main__":
    print("Testing issue #306: Module-level allocatable arrays")
    print("=" * 60)

    test_allocate_and_access()
    test_reallocate_different_size()
    test_multiple_reallocations()

    print("\n" + "=" * 60)
    print("All tests passed!")
