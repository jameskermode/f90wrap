import unittest

import pywrapper


class TestPolymorphicGlobal(unittest.TestCase):
    def test_plain_type_global(self):
        t = pywrapper.main.global_testtype
        t.value = 1.5
        self.assertAlmostEqual(pywrapper.main.global_testtype.value, 1.5)

    def test_class_global(self):
        c = pywrapper.main.global_testclass
        c.value = 3.0
        self.assertAlmostEqual(pywrapper.main.global_testclass.value, 3.0)

    def test_class_global_method(self):
        c = pywrapper.main.global_testclass
        c.value = 4.0
        self.assertAlmostEqual(c.method(), 8.0)

    def test_class_global_setter(self):
        c = pywrapper.main.testclass()
        c.value = 5.0
        pywrapper.main.global_testclass = c
        self.assertAlmostEqual(pywrapper.main.global_testclass.value, 5.0)


if __name__ == "__main__":
    unittest.main()
