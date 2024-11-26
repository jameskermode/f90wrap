#!/usr/bin/env python
import pytest
from itest.myclass import get_create_count, get_destroy_count
from itest.myclass_factory import myclass_create

REF = 3.1415
TOL = 1.0e-6

def test_create_destroy_object():
    assert get_create_count() == 0
    assert get_destroy_count() == 0

    obj = myclass_create(REF)

    assert get_create_count() == 1
    assert get_destroy_count() == 0

    assert abs(obj.get_val() - REF) < TOL

    del obj

    assert get_create_count() == 1
    assert get_destroy_count() == 1

if __name__ == '__main__':
    pytest.main([__file__])
