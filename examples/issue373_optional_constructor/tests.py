import unittest

from pywrapper import widget_mod


class TestOptionalConstructor(unittest.TestCase):
    def test_constructor_without_optional(self):
        w = widget_mod.widget_t(3.0)
        self.assertAlmostEqual(w.x, 3.0)
        self.assertAlmostEqual(w.scale, 1.0)

    def test_constructor_with_optional(self):
        w = widget_mod.widget_t(2.0, scale=0.5)
        self.assertAlmostEqual(w.x, 2.0)
        self.assertAlmostEqual(w.scale, 0.5)


if __name__ == "__main__":
    unittest.main()
