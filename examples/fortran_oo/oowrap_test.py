import unittest
import math
import numpy as np

from pywrapper import m_geometry

circle_radius = 1.5
ball_radius = 2.5
square_size = 3.
precision_4 = 1e-6
precision_8 = 1e-12


class TestsGlobal(unittest.TestCase):
    def test_pi_4(self):
        try:
            fortran_pi = m_geometry.pi
        except AttributeError:
            fortran_pi = m_geometry.get_pi()  # f90wrap package flag enable
        self.assertAlmostEqual(fortran_pi, math.pi, delta=precision_8*math.pi)

    def test_pi_8(self):
        try:
            fortran_pi = m_geometry.pi
        except AttributeError:
            fortran_pi = m_geometry.get_pi()  # f90wrap package flag enable
        self.assertAlmostEqual(fortran_pi, math.pi, delta=precision_8*math.pi)

class TestsCircle(unittest.TestCase):
    def setUp(self):
        self.circle = m_geometry.Circle(circle_radius, ball_radius)

    def test_explicit_constructor(self):
        self.assertIsInstance(self.circle, m_geometry.Circle)
        self.assertEqual(self.circle.radius, circle_radius)

    @unittest.skip("Support for this feature is not planned for now")
    def test_implicit_constructor(self):
        circle = m_geometry.Circle(circle_radius)
        self.assertIsInstance(circle, m_geometry.Circle)
        self.assertEqual(circle.radius, circle_radius)

    def test_has_member(self):
        self.assertTrue(hasattr(self.circle, 'radius'))

    def test_has_public_method(self):
        self.assertTrue(hasattr(self.circle, 'area'))
        self.assertTrue(hasattr(self.circle, 'print'))

    def test_has_private_method(self):
        self.assertTrue(hasattr(self.circle, 'private_method'))

    def test_setter(self):
        new_radius = 3.7
        self.circle.radius = new_radius
        try:
            f_radius = m_geometry.get_circle_radius(self.circle)
        except AttributeError:
            f_radius = self.circle.get_circle_radius()
        self.assertAlmostEqual(f_radius, new_radius, delta=precision_8*new_radius)

    @unittest.skip("Support for this feature is not planned for now")
    def test_area_proc(self):
        self.circle.radius = circle_radius
        py_area = math.pi * circle_radius**2
        self.assertAlmostEqual(
            m_geometry.circle_area(self.circle), py_area, delta=precision_8*py_area)

    def test_copy(self):
        new_radius = self.circle.radius * 2
        from_circle = m_geometry.Circle(new_radius, ball_radius)
        self.circle.copy(from_circle)
        self.assertEqual(self.circle.radius, from_circle.radius)


class TestsInheritance(unittest.TestCase):
    def setUp(self):
        self.ball = m_geometry.Ball(circle_radius, ball_radius)

    def test_inheritance(self):
        self.assertIsInstance(self.ball, m_geometry.Circle)

    def test_inheritance_member(self):
        self.assertTrue(hasattr(self.ball, 'radius'))

    def test_inheritance_method(self):
        self.assertTrue(hasattr(self.ball, 'print'))


class TestsBall(unittest.TestCase):
    def setUp(self):
        self.ball = m_geometry.Ball(circle_radius, ball_radius)

    def test_explicit_constructor(self):
        self.assertIsInstance(self.ball, m_geometry.Ball)
        self.assertEqual(self.ball.radius, ball_radius)

    @unittest.skip("Support for this feature is not planned for now")
    def test_implicit_constructor(self):
        ball = m_geometry.Ball(ball_radius)
        self.assertIsInstance(ball, m_geometry.Ball)
        self.assertEqual(ball.radius, ball_radius)

    def test_has_public_method(self):
        self.assertTrue(hasattr(self.ball, 'area'))
        self.assertTrue(hasattr(self.ball, 'volume'))

    def test_has_private_method(self):
        self.assertTrue(hasattr(self.ball, 'private_method'))

    def test_setter(self):
        new_radius = 3.7
        self.ball.radius = new_radius
        try:
            f_radius = m_geometry.get_ball_radius(self.ball)
        except AttributeError:
            f_radius = self.ball.get_ball_radius()
        self.assertAlmostEqual(f_radius, new_radius, delta=precision_8*new_radius)

    @unittest.skip("Support for this feature is not planned for now")
    def test_area_proc(self):
        self.ball.radius = ball_radius
        py_area = 4 * math.pi * ball_radius**2
        self.assertAlmostEqual(m_geometry.circle_area(self.ball), py_area, delta=precision_8*py_area)


class TestsPolymorphism(unittest.TestCase):
    def setUp(self):
        self.ball = m_geometry.Ball(circle_radius, ball_radius)
        self.circle = m_geometry.Circle(circle_radius, ball_radius)

    def test_polymorphism(self):
        self.ball.radius = circle_radius
        try:
            f_radius = m_geometry.get_circle_radius(self.ball)
        except AttributeError:
            f_radius = self.ball.get_circle_radius()
        self.assertEqual(f_radius, circle_radius)

    def test_bad_polymorphism(self):
        self.circle.radius = circle_radius
        with self.assertRaises(TypeError):
            try:
                m_geometry.get_ball_radius(self.circle)
            except AttributeError:
                raise TypeError

    def test_bad_polymorphism_w_move(self):
        self.circle.radius = circle_radius
        with self.assertRaises(AttributeError):
            self.circle.get_ball_radius()


class TestsSpecificBinding(unittest.TestCase):
    def setUp(self):
        self.circle = m_geometry.Circle(circle_radius, ball_radius)
        self.ball = m_geometry.Ball(circle_radius, ball_radius)

    def test_call_circle(self):
        self.circle.print()

    def test_call_circle_2(self):
        self.circle.obj_name()

    def test_bad_call_circle(self):
        with self.assertRaises(AttributeError):
            self.circle.circle_print()

    def test_area_circle(self):
        self.circle.radius = circle_radius
        py_area = math.pi * circle_radius**2
        self.assertAlmostEqual(self.circle.area(), py_area, delta=precision_8*py_area)

    def test_call_ball(self):
        self.ball.print()

    def test_bad_call_ball(self):
        with self.assertRaises(AttributeError):
            self.ball.ball_print()

    def test_area_ball(self):
        self.ball.radius = ball_radius
        py_area = 4 * math.pi * ball_radius**2
        self.assertAlmostEqual(self.ball.area(), py_area, delta=precision_8*py_area)

    def test_volume_ball(self):
        self.ball.radius = ball_radius
        py_volume = 4.0 / 3.0 * math.pi * ball_radius**3
        self.assertAlmostEqual(self.ball.volume(), py_volume, delta=precision_8*py_volume)


class TestsGenericBinding(unittest.TestCase):
    def setUp(self):
        self.circle = m_geometry.Circle(circle_radius, ball_radius)

    def test_perimeter_4(self):
        radius = 4.2
        py_perimeter = 2.0 * math.pi * radius
        f_perimeter = self.circle.perimeter_4(np.float32(radius))
        self.assertAlmostEqual(f_perimeter, py_perimeter, delta=precision_4*py_perimeter)
        self.assertNotAlmostEqual(f_perimeter, py_perimeter, delta=precision_8*py_perimeter)

    def test_perimeter_8(self):
        radius = 4.2
        py_perimeter = 2.0 * math.pi * radius
        f_perimeter = self.circle.perimeter_8(np.float64(radius))
        self.assertAlmostEqual(f_perimeter, py_perimeter, delta=precision_8*py_perimeter)

    def test_perimeter_4_poly(self):
        radius = 4.2
        py_perimeter = 2.0 * math.pi * radius
        f_perimeter = self.circle.perimeter(np.float32(radius))
        self.assertAlmostEqual(f_perimeter, py_perimeter, delta=precision_4*py_perimeter)
        self.assertNotAlmostEqual(f_perimeter, py_perimeter, delta=precision_8*py_perimeter)

    def test_perimeter_8_poly(self):
        radius = 4.2
        py_perimeter = 2.0 * math.pi * radius
        f_perimeter = self.circle.perimeter(np.float64(radius))
        self.assertAlmostEqual(f_perimeter, py_perimeter, delta=precision_8*py_perimeter)

class TestsAbstractType(unittest.TestCase):
    def setUp(self):
        self.square = m_geometry.Square(square_size)

    def test_init_abstract(self):
        with self.assertRaises(NotImplementedError):
            m_geometry.Rectangle()

    def test_init_child(self):
        self.assertIsInstance(self.square, m_geometry.Square)
        self.assertIsInstance(self.square, m_geometry.Rectangle)

    def test_getter(self):
        self.assertEqual(self.square.length, square_size)
        self.assertEqual(self.square.width, square_size)

    def test_specific_method(self):
        py_perimeter = square_size * 4
        f_perimeter = self.square.perimeter()
        self.assertAlmostEqual(f_perimeter, py_perimeter, delta=precision_8*py_perimeter)

    def test_specific_method_overload(self):
        self.assertEqual(self.square.is_square(), 1)

    def test_multi_level_abstract(self):
        self.assertEqual(self.square.is_polygone(), 1)

    def test_deferred_method(self):
        py_area = square_size**2
        f_area = self.square.area()
        self.assertAlmostEqual(f_area, py_area, delta=precision_8*py_area)

    def test_setter(self):
        new_size = 3.6
        self.square.length = new_size
        self.square.width = new_size
        self.assertAlmostEqual(self.square.length, new_size, delta=precision_8*new_size)
        self.assertAlmostEqual(self.square.width, new_size, delta=precision_8*new_size)

    def test_compare(self):
        square1 = m_geometry.Square(square_size)
        square2 = m_geometry.Square(square_size)
        self.assertTrue(square1.is_equal(square2))

    def test_return_class(self):
        square1 = m_geometry.Square(square_size)
        diamond = square1.create_diamond()
        self.assertEqual(diamond.width, square_size)
        self.assertEqual(diamond.length, square_size)

class TestsListClass(unittest.TestCase):
    def test_list_square(self):
        n_square = 5
        width=10.
        list_square = m_geometry.List_square()
        list_square.init(n_square)
        for i in range(n_square):
            list_square.alloc_class[i].init(width+i)
            list_square.ptr_class[i].init(width+i)
            list_square.alloc_type[i].init(width+i)
            list_square.ptr_type[i].init(width+i)
        list_square.scalar_type.init(width)
        list_square.scalar_class.init(width)

        # Test getter of list of derived types elements
        for i in range(n_square):
            ref = m_geometry.Square(width+i)
            self.assertTrue(list_square.alloc_class[i].is_equal(ref))
            self.assertTrue(list_square.ptr_class[i].is_equal(ref))
            self.assertTrue(list_square.alloc_type[i].is_equal(ref))
            self.assertTrue(list_square.ptr_type[i].is_equal(ref))
        ref = m_geometry.Square(width)
        self.assertTrue(list_square.scalar_type.is_equal(ref))
        self.assertTrue(list_square.scalar_class.is_equal(ref))

        # Test setter of list of derived types elements
        for i in range(n_square):
            ref = m_geometry.Square(width-i)
            list_square.alloc_class[i] = ref
            list_square.ptr_class[i] = ref
            list_square.alloc_type[i] = ref
            list_square.ptr_type[i] = ref
            self.assertTrue(list_square.alloc_class[i].is_equal(ref))
            self.assertTrue(list_square.ptr_class[i].is_equal(ref))
            self.assertTrue(list_square.alloc_type[i].is_equal(ref))
            self.assertTrue(list_square.ptr_type[i].is_equal(ref))
        ref = m_geometry.Square(100.)
        list_square.scalar_type = ref
        list_square.scalar_class = ref
        self.assertTrue(list_square.scalar_type.is_equal(ref))
        self.assertTrue(list_square.scalar_class.is_equal(ref))

    def test_list_circle(self):
        n_circle = 5
        width=10.
        list_circle = m_geometry.List_circle()
        list_circle.init(n_circle)
        for i in range(n_circle):
            list_circle.alloc_class[i].init(width+i)
            list_circle.ptr_class[i].init(width+i)
            list_circle.alloc_type[i].init(width+i)
            list_circle.ptr_type[i].init(width+i)
        list_circle.scalar_type.init(width)
        list_circle.scalar_class.init(width)

        # Test getter of list of derived types elements
        for i in range(n_circle):
            ref = m_geometry.Circle(width+i, width+i)
            self.assertEqual(list_circle.alloc_class[i].radius, ref.radius)
            self.assertEqual(list_circle.ptr_class[i].radius, ref.radius)
            self.assertEqual(list_circle.alloc_type[i].radius, ref.radius)
            self.assertEqual(list_circle.ptr_type[i].radius, ref.radius)
        ref = m_geometry.Circle(width, width)
        self.assertEqual(list_circle.scalar_type.radius, ref.radius)
        self.assertEqual(list_circle.scalar_class.radius, ref.radius)

        # Test setter of list of derived types elements
        for i in range(n_circle):
            ref = m_geometry.Circle(width-i, width-i)
            # Class Circle has no assignment(=) method thus
            # we expect setitem setter to be undefined
            with self.assertRaises(TypeError):
                list_circle.alloc_class[i] = ref
            with self.assertRaises(TypeError):
                list_circle.ptr_class[i] = ref
            list_circle.alloc_type[i] = ref
            list_circle.ptr_type[i] = ref
            self.assertEqual(list_circle.alloc_type[i].radius, ref.radius)
            self.assertEqual(list_circle.ptr_type[i].radius, ref.radius)
        ref = m_geometry.Circle(100., 100.)
        list_circle.scalar_type = ref
        with self.assertRaises(AttributeError):
            list_circle.scalar_class = ref
        self.assertEqual(list_circle.scalar_type.radius, ref.radius)

    def test_array(self):
        a = m_geometry.Array_3d()
        n1=10
        n2=2
        n3=3
        a.init_3d(n1,n2,n3)
        shape = a.values_3d.shape
        self.assertEqual(shape[0], n1)
        self.assertEqual(shape[1], n2)
        self.assertEqual(shape[2], n3)
        shape = a.values.shape
        print(shape)
        self.assertEqual(shape[0], n1*n2*n3)
        shape = a.buf.shape
        self.assertEqual(shape[0], n1*n2*n3)
        shape = a.values_3d.shape
        self.assertEqual(shape[0], n1)
        self.assertEqual(shape[1], n2)
        self.assertEqual(shape[2], n3)
        shape = a.buf.shape
        self.assertEqual(shape[0], n1*n2*n3)
        shape = a.values.shape
        self.assertEqual(shape[0], n1*n2*n3)

if __name__ == '__main__':
    unittest.main()
