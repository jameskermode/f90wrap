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
        gc.collect()
        t = []
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        for i in range(8192):
            t.append(itest.alloc_output.alloc_output_type_func(VAL))
        del t
        gc.collect()
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        self.assertLess(sum(stat.size_diff for stat in stats), 4096)

if __name__ == '__main__':
    unittest.main()
