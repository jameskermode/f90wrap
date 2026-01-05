#!/usr/bin/env python
"""Test for issue #357: module variable/argument name conflict."""
import base

# Test that the function works correctly
result = base.base.a_times_b_plus_c(2.0, 3.0, 4.0)
expected = 2.0 * 3.0 + 4.0  # = 10.0
assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

print("Issue #357 test passed: a_times_b_plus_c(2, 3, 4) =", result)
