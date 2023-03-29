import unittest
import library
lib = library.test


class TestExample(unittest.TestCase):

    def setUp(self):
        pass

    def test_create(self):
        data = lib.atype()
        lib.create(data, 10)
        self.assertEqual(len(data.array), 10)

    def test_type_create(self):
        data = lib.atype()
        data.p_create(10)
        self.assertEqual(len(data.array), 10)

    def test_asum(self):
        data = lib.atype()
        lib.create(data, 10)
        data.array[:] = 1
        a = lib.asum(data)
        self.assertEqual(a, 10)
        a = lib.asum_class(data)
        self.assertEqual(a, 10)

    def test_type_asum(self):
        data = lib.atype()
        data.p_create(10)
        data.array[:] = 1
        a = data.p_asum()
        self.assertEqual(a, 10)
        a2 = data.asum_class()
        self.assertEqual(a2, 10)
        a3 = data.p_asum_2()
        self.assertEqual(a3, 10)

    def test_type_asum_b(self):
        data = lib.atype()
        data.p_create(10)
        data.array[:] = 1
        a = data.p_asum()
        self.assertEqual(a, 10)
        data2 = lib.btype()
        data2.array[:] = 1
        b = data2.p_asum()
        self.assertEqual(b, 3)

    def test_type_assignment(self):
        data = lib.atype()
        data.p_create(10)
        data.array[:] = 1
        data.p_reset(2)
        a = data.p_asum()
        self.assertEqual(a, 20)


if __name__ == '__main__':

    unittest.main()
