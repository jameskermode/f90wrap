#!/usr/bin/env python
import unittest
import gc
import tracemalloc
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
        # Run multiple rounds to detect real memory leaks vs one-time overhead.
        # Python's weakref.finalize uses a dict that doesn't shrink after pop(),
        # but the memory IS reused. A real leak would show growth between rounds.
        num_objects = 8192

        def run_round():
            t = []
            for i in range(num_objects):
                t.append(itest.alloc_output.alloc_output_type_func(VAL))
            del t
            gc.collect()

        # Round 1: warm up (allocates dict space)
        gc.collect()
        run_round()

        # Round 2: measure baseline after warmup
        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()
        run_round()
        after_round2 = tracemalloc.take_snapshot()

        # Round 3: check for growth
        run_round()
        after_round3 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Memory should not grow significantly between rounds 2 and 3
        # (allowing small overhead for tracemalloc itself)
        diff_r2 = sum(s.size_diff for s in after_round2.compare_to(baseline, 'lineno'))
        diff_r3 = sum(s.size_diff for s in after_round3.compare_to(after_round2, 'lineno'))

        # Round 2 may have some growth from tracemalloc overhead, but round 3
        # should show no significant growth - a real leak would grow each round
        self.assertLess(diff_r3, 4096, f"Memory grew between rounds: {diff_r3} bytes")

if __name__ == '__main__':
    unittest.main()
