#!/usr/bin/env python
import itest


def main():
    test_type_output_is_wrapped()
    test_intrinsic_output_is_not_wrapped()
    test_array_output_is_not_wrapped()
    test_type_output_wrapper()


def test_type_output_is_wrapped():
    assert hasattr(itest.alloc_output, 'alloc_output_type_func')


def test_intrinsic_output_is_not_wrapped():
    assert (not hasattr(itest.alloc_output, 'alloc_output_intrinsic_func'))


def test_array_output_is_not_wrapped():
    assert (not hasattr(itest.alloc_output, 'alloc_output_array_func'))


def test_type_output_wrapper():
    VAL = 10.0
    TOL = 1e-13

    t = itest.alloc_output.alloc_output_type_func(VAL)
    assert(abs(t.a - VAL) < TOL)

if __name__ == '__main__':
    main()
