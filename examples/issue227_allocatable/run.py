#!/usr/bin/env python
import os
import gc
import tracemalloc

import itest


def main():
    test_type_output_is_wrapped()
    test_intrinsic_output_is_not_wrapped()
    test_array_output_is_not_wrapped()
    test_type_output_wrapper()
    test_memory_leak()


def test_type_output_is_wrapped():
    assert hasattr(itest.alloc_output, 'alloc_output_type_func')


def test_intrinsic_output_is_not_wrapped():
    assert (not hasattr(itest.alloc_output, 'alloc_output_intrinsic_func'))


def test_array_output_is_not_wrapped():
    assert (not hasattr(itest.alloc_output, 'alloc_output_array_func'))


VAL = 10.0
TOL = 1e-13


def test_type_output_wrapper():
    t = itest.alloc_output.alloc_output_type_func(VAL)
    assert(abs(t.a - VAL) < TOL)


def test_memory_leak():
    gc.collect()
    t = []
    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    for i in range(2048):
        t.append(itest.alloc_output.alloc_output_type_func(VAL))
    del t
    gc.collect()
    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    stats = end_snapshot.compare_to(start_snapshot, 'lineno')
    assert sum(stat.size_diff for stat in stats) < 1024


if __name__ == '__main__':
    main()
