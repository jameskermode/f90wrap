#!/usr/bin/env python
import unittest
import gc
import re
import os

import itest

VAL = 10.0
TOL = 1e-13

class TestAllocOutput(unittest.TestCase):

    def test_type_output_is_wrapped(self):
        self.assertTrue(hasattr(itest.alloc_output, 'alloc_output_type_func'))

    def test_intrinsic_output_is_not_wrapped(self):
        self.assertFalse(hasattr(itest.alloc_output, 'alloc_output_intrinsic_func'))

    def test_array_output_is_not_wrapped(self):
        self.assertFalse(hasattr(itest.alloc_output, 'alloc_output_array_func'))

    def test_type_output_wrapper(self):
        t = itest.alloc_output.alloc_output_type_func(VAL)
        self.assertAlmostEqual(t.a, VAL, delta=TOL)

    @unittest.skipIf(re.search("nvfortran", os.environ.get('F90', '')), "Fails with nvfortran")
    def test_memory_leak(self):
        # Test for memory leaks by verifying all finalizers are called.
        # This directly tests the memory management rather than relying on
        # tracemalloc which can be flaky due to Python runtime variations.
        num_objects = 100
        call_count = [0]
        original_finalise = itest._itest.f90wrap_alloc_output__alloc_output_type_finalise

        def counting_finalise(handle):
            call_count[0] += 1
            return original_finalise(handle)

        try:
            itest._itest.f90wrap_alloc_output__alloc_output_type_finalise = counting_finalise
            gc.collect()
            call_count[0] = 0

            # Create objects
            objs = [itest.alloc_output.alloc_output_type_func(VAL) for _ in range(num_objects)]
            self.assertEqual(call_count[0], 0, "Finalizer called prematurely")

            # Delete objects and force GC
            del objs
            gc.collect()

            # All finalizers should have been called
            self.assertEqual(call_count[0], num_objects,
                f"Expected {num_objects} finalizer calls, got {call_count[0]}")
        finally:
            itest._itest.f90wrap_alloc_output__alloc_output_type_finalise = original_finalise

    @unittest.skipIf(re.search("nvfortran", os.environ.get('F90', '')), "Fails with nvfortran")
    @unittest.skipIf(os.name == 'nt', "resource module not available on Windows")
    def test_memory_stability(self):
        # Test that memory usage stabilizes over multiple rounds.
        # A real memory leak would show continuous growth; normal behavior
        # shows initial growth that stabilizes as allocator reuses memory.
        import sys
        import resource
        num_objects = 4096
        num_rounds = 10

        def run_round():
            t = [itest.alloc_output.alloc_output_type_func(VAL) for _ in range(num_objects)]
            del t
            gc.collect()

        def get_rss_kb():
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS returns bytes, Linux returns KB
            if sys.platform == 'darwin':
                rss = rss // 1024
            return rss

        # Warmup
        gc.collect()
        for _ in range(3):
            run_round()

        gc.collect()
        rss_after_warmup = get_rss_kb()

        # Run multiple rounds
        for _ in range(num_rounds):
            run_round()

        gc.collect()
        rss_final = get_rss_kb()

        # Check growth - a real leak with 4096*10=40960 objects would leak MB
        # Allow 1MB growth for normal allocator overhead
        growth_kb = rss_final - rss_after_warmup
        self.assertLess(growth_kb, 1024,
            f"RSS grew by {growth_kb} KB over {num_rounds} rounds - possible leak")

if __name__ == '__main__':
    unittest.main()
