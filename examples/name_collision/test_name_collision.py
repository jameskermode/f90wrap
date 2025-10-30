#!/usr/bin/env python3
"""
Test script to verify that argument name collision resolution works correctly.

This test ensures that when an argument name conflicts with a USE-imported symbol,
f90wrap automatically renames the argument in both the Fortran wrapper and the
Python wrapper, preventing both compilation errors and runtime TypeErrors.
"""
import sys
import test_collision

def test_system_init_with_renamed_arg():
    """
    Test that system_init can be called despite enable_timing name collision.

    The function has an argument named 'enable_timing' which conflicts with
    a subroutine of the same name in the module. f90wrap should automatically
    rename this to 'enable_timing_in' in the wrappers.
    """
    print("Testing system_init with enable_timing_in=True...")
    try:
        # The argument was renamed from enable_timing to enable_timing_in to avoid collision
        test_collision.test_module.system_init(enable_timing_in=True, verbosity=2)
        print("✓ system_init(enable_timing_in=True) succeeded")
    except TypeError as e:
        print(f"✗ FAILED: {e}")
        print("This indicates the argument renaming didn't work correctly")
        return False

    print("\nTesting system_init with enable_timing_in=False...")
    try:
        test_collision.test_module.system_init(enable_timing_in=False)
        print("✓ system_init(enable_timing_in=False) succeeded")
    except TypeError as e:
        print(f"✗ FAILED: {e}")
        return False

    return True

def test_enable_timing_function():
    """Test that the enable_timing subroutine can still be called."""
    print("\nTesting enable_timing subroutine...")
    try:
        test_collision.test_module.enable_timing()
        print("✓ enable_timing() subroutine call succeeded")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    return True

def main():
    print("=" * 60)
    print("Name Collision Test Suite")
    print("=" * 60)
    print()

    all_passed = True

    # Test the main collision scenario
    if not test_system_init_with_renamed_arg():
        all_passed = False

    # Test that the original function is still accessible
    if not test_enable_timing_function():
        all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("✓ All tests PASSED")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests FAILED")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
