#!/usr/bin/env python
import unittest
from itest.mytype import mytype_create, mytype_destroy
from itest.myclass_factory import myclass_create

REF = 3.1415
TOL = 1.0e-6

class TestMyType(unittest.TestCase):

    def test_create_destroy_type_object(self):
        """Object creation and destruction should happen only once."""
        from itest.mytype import (
            get_create_count,
            get_destroy_count,
            set_create_count,
            set_destroy_count,
        )
        set_create_count(0)
        set_destroy_count(0)

        obj = mytype_create(REF)

        self.assertEqual(get_create_count(), 1)
        self.assertEqual(get_destroy_count(), 0)

        self.assertTrue(abs(obj.val - REF) < TOL)

        del obj

        self.assertEqual(get_create_count(), 1)
        self.assertEqual(get_destroy_count(), 1)

    def test_type_member_access(self):
        """Direct access of member variables."""
        obj = mytype_create(REF)

        self.assertTrue(abs(obj.val - REF) < TOL)

        obj.val = 2.0 * REF

        self.assertTrue(abs(obj.val - 2.0 * REF) < TOL)

        del obj


class TestMyClass(unittest.TestCase):

    def test_create_destroy_class_object(self):
        """Object creation and destruction should happen only once."""
        from itest.myclass import (
            get_create_count,
            get_destroy_count,
            set_create_count,
            set_destroy_count,
        )
        set_create_count(0)
        set_destroy_count(0)

        obj = myclass_create(REF)

        self.assertEqual(get_create_count(), 1)
        self.assertEqual(get_destroy_count(), 0)

        self.assertTrue(abs(obj.get_val() - REF) < TOL)

        del obj

        self.assertEqual(get_create_count(), 1)
        self.assertEqual(get_destroy_count(), 1)

    def test_class_getter_setter(self):
        """Getters and setters defined in Fortran should work."""
        obj = myclass_create(REF)

        self.assertTrue(abs(obj.get_val() - REF) < TOL)

        obj.set_val(2.0 * REF)

        self.assertTrue(abs(obj.get_val() - 2.0 * REF) < TOL)

        del obj

    def test_class_member_access(self):
        """Direct access of member variables."""
        obj = myclass_create(REF)

        self.assertTrue(abs(obj.val - REF) < TOL)

        obj.val = 2.0 * REF

        self.assertTrue(abs(obj.val - 2.0 * REF) < TOL)

        del obj


if __name__ == "__main__":
    unittest.main()
