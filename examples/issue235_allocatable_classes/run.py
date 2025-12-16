#!/usr/bin/env python
import gc
import unittest
from itest import mytype, myclass, myclass_factory

REF = 3.1415
TOL = 1.0e-6

class TestMyType(unittest.TestCase):

    def test_create_destroy_type_object(self):
        """Object creation and destruction should happen only once."""
        mytype.create_count = 0
        mytype.destroy_count = 0

        obj = mytype.mytype_create(REF)

        self.assertEqual(mytype.create_count, 1)

        self.assertTrue(abs(obj.val - REF) < TOL)

        del obj
        gc.collect()

        self.assertEqual(mytype.create_count, 1)
        self.assertGreaterEqual(mytype.destroy_count, 1)

    def test_type_member_access(self):
        """Direct access of member variables."""
        obj = mytype.mytype_create(REF)

        self.assertTrue(abs(obj.val - REF) < TOL)

        obj.val = 2.0 * REF

        self.assertTrue(abs(obj.val - 2.0 * REF) < TOL)

        del obj


class TestMyClass(unittest.TestCase):

    def test_create_destroy_class_object(self):
        """Object creation and destruction should happen only once."""
        myclass.create_count = 0
        myclass.destroy_count = 0

        obj = myclass_factory.myclass_create(REF)

        self.assertEqual(myclass.create_count, 1)

        self.assertTrue(abs(obj.get_val() - REF) < TOL)

        del obj
        gc.collect()

        self.assertEqual(myclass.create_count, 1)
        self.assertGreaterEqual(myclass.destroy_count, 1)

    def test_class_getter_setter(self):
        """Getters and setters defined in Fortran should work."""
        obj = myclass_factory.myclass_create(REF)

        self.assertTrue(abs(obj.get_val() - REF) < TOL)

        obj.set_val(2.0 * REF)

        self.assertTrue(abs(obj.get_val() - 2.0 * REF) < TOL)

        del obj

    def test_class_member_access(self):
        """Direct access of member variables."""
        obj = myclass_factory.myclass_create(REF)

        self.assertTrue(abs(obj.val - REF) < TOL)

        obj.val = 2.0 * REF

        self.assertTrue(abs(obj.val - 2.0 * REF) < TOL)

        del obj


if __name__ == "__main__":
    unittest.main()
