import pytest
import math
import numpy as np

from pywrapper import m_geometry

circle_radius = 1.5
ball_radius = 2.5
square_size = 3.
precision = 1e-7

@pytest.fixture(scope="function")
def circle():
  return m_geometry.Circle(circle_radius, ball_radius)

@pytest.fixture(scope="function")
def ball():
  return m_geometry.Ball(circle_radius, ball_radius)

@pytest.fixture(scope="function")
def square():
  return m_geometry.Square(square_size)

class Tests_Global():
  @pytest.mark.dependency()
  def test_pi_4(self, circle):
    try:
      fortran_pi = m_geometry.pi
    except AttributeError:
      fortran_pi = m_geometry.get_pi() # f90wrap package flag enable
    assert fortran_pi == pytest.approx(math.pi, rel=precision)

  @pytest.mark.dependency()
  def test_pi_8(self, circle):
    try:
      fortran_pi = m_geometry.pi
    except AttributeError:
      fortran_pi = m_geometry.get_pi() # f90wrap package flag enable
    assert fortran_pi == pytest.approx(math.pi, rel=precision**2)

class Tests_Circle():
  def test_explicit_constructor(self, circle):
    assert isinstance(circle, m_geometry.Circle)
    assert circle.radius == circle_radius

  @pytest.mark.skip(reason="Support for this feature is not planned for now")
  def test_implicit_constructor(self):
    circle = m_geometry.Circle(circle_radius)
    assert isinstance(circle, m_geometry.Circle)
    assert circle.radius == circle_radius

  @pytest.mark.dependency()
  def test_has_member(self, circle):
    assert hasattr(circle, 'radius')

  def test_has_public_method(self, circle):
    assert hasattr(circle, 'area')
    assert hasattr(circle, 'print')

  def test_has_private_method(self, circle):
    assert hasattr(circle, 'private_method')

  @pytest.mark.dependency()
  @pytest.mark.dependency(depends=["Tests_Circle::test_has_member"])
  def test_setter(self, circle):
    new_radius = 3.7
    circle.radius = new_radius
    try:
      f_radius = m_geometry.get_circle_radius(circle)
    except AttributeError:
      f_radius = circle.get_circle_radius()

    assert f_radius == pytest.approx(new_radius, rel=precision)

  @pytest.mark.dependency(depends=["Tests_Circle::test_setter",
                                  "Tests_Global::test_pi_4"])
  @pytest.mark.skip(reason="Support for this feature is not planned for now")
  def test_area_proc(self, circle):
    circle.radius = circle_radius
    py_area = math.pi*circle_radius**2
    assert m_geometry.circle_area(circle) == pytest.approx(py_area, rel=precision)

  @pytest.mark.dependency(depends=["Tests_Circle::test_setter"])
  def test_copy(self, circle):
    new_radius = circle.radius*2
    from_circle = m_geometry.Circle(new_radius, ball_radius)
    circle.copy(from_circle)
    assert circle.radius == from_circle.radius


@pytest.mark.dependency()
class Tests_inheritance():
  def test_inheritance(self, ball):
    assert isinstance(ball, m_geometry.Circle)

  @pytest.mark.dependency()
  def test_inheritance_member(self, ball):
    assert hasattr(ball, 'radius')

  @pytest.mark.dependency()
  def test_inheritance_method(self, ball):
    assert hasattr(ball, 'print')

class Tests_Ball():
  @pytest.mark.dependency(depends=["Tests_inheritance::test_inheritance_member"])
  def test_explicit_constructor(self, ball):
    assert isinstance(ball, m_geometry.Ball)
    assert ball.radius == ball_radius

  @pytest.mark.dependency(depends=["Tests_inheritance::test_inheritance_member"])
  @pytest.mark.skip(reason="Support for this feature is not planned for now")
  def test_implicit_constructor(self):
    ball = m_geometry.Ball(ball_radius)
    assert isinstance(ball, m_geometry.Ball)
    assert ball.radius == ball_radius

  def test_has_public_method(self, ball):
    assert hasattr(ball, 'area')
    assert hasattr(ball, 'volume')

  def test_has_private_method(self, ball):
    assert hasattr(ball, 'private_method')

  @pytest.mark.dependency()
  @pytest.mark.dependency(depends=["Tests_inheritance::test_inheritance_member"])
  def test_setter(self, ball):
    new_radius = 3.7
    ball.radius = new_radius
    try:
      f_radius = m_geometry.get_ball_radius(ball)
    except AttributeError:
      f_radius = ball.get_ball_radius()
    assert f_radius == pytest.approx(new_radius, rel=precision)

  @pytest.mark.dependency(depends=["Tests_Ball::test_setter",
                                  "Tests_Global::test_pi_4"])
  @pytest.mark.skip(reason="Support for this feature is not planned for now")
  def test_area_proc(self, ball):
    ball.radius = ball_radius
    py_area = 4*math.pi*ball_radius**2
    assert m_geometry.circle_area(ball) == pytest.approx(py_area, rel=precision)

class Tests_polymorphism():
  def test_polymorphism(self, ball):
    ball.radius = circle_radius
    try:
      f_radius = m_geometry.get_circle_radius(ball)
    except AttributeError:
      f_radius = ball.get_circle_radius()
    assert f_radius == circle_radius

  def test_bad_polymorphism(self, circle):
    circle.radius = circle_radius
    with pytest.raises(TypeError):
      try:
        m_geometry.get_ball_radius(circle)
      except AttributeError:
        raise(TypeError)

  def test_bad_polymorphism_w_move(self, circle):
    circle.radius = circle_radius
    with pytest.raises(AttributeError):
      circle.get_ball_radius()

class Tests_specific_binding():
  def test_call_circle(self, circle):
    circle.print()

  def test_call_circle_2(self, circle):
    circle.obj_name()

  def test_bad_call_circle(self, circle):
    with pytest.raises(AttributeError):
      circle.circle_print()

  @pytest.mark.dependency(depends=["Tests_Circle::test_setter",
                                  "Tests_Global::test_pi_4"])
  def test_area_circle(self, circle):
    circle.radius = circle_radius
    py_area = math.pi*circle_radius**2
    assert circle.area() == pytest.approx(py_area, rel=precision)

  @pytest.mark.dependency(depends=["Tests_inheritance::test_inheritance_member"])
  def test_call_ball(self, ball):
    ball.print()

  def test_bad_call_ball(self, ball):
    with pytest.raises(AttributeError):
      ball.ball_print()

  @pytest.mark.dependency(depends=["Tests_Ball::test_setter",
                                  "Tests_Global::test_pi_4"])
  def test_area_ball(self, ball):
    ball.radius = ball_radius
    py_area = 4*math.pi*ball_radius**2
    assert ball.area() == pytest.approx(py_area, rel=precision)

  @pytest.mark.dependency(depends=["Tests_Ball::test_setter",
                                  "Tests_Global::test_pi_4"])
  def test_volume_ball(self, ball):
    ball.radius = ball_radius
    py_volume = 4./3.*math.pi*ball_radius**3
    assert ball.volume() == pytest.approx(py_volume, rel=precision)

class Tests_generic_binding():
  @pytest.mark.dependency(depends=["Tests_Global::test_pi_8"])
  def test_perimeter_4(self, circle):
    radius = 4.2
    py_perimeter = 2.*math.pi*radius
    f_perimeter = circle.perimeter_4(np.float32(radius))
    assert f_perimeter == pytest.approx(py_perimeter, rel=precision)
    assert f_perimeter != pytest.approx(py_perimeter, rel=precision**2)

  @pytest.mark.dependency(depends=["Tests_Global::test_pi_8"])
  def test_perimeter_8(self, circle):
    radius = 4.2
    py_perimeter = 2.*math.pi*radius
    f_perimeter = circle.perimeter_8(np.float64(radius))
    assert f_perimeter == pytest.approx(py_perimeter, rel=precision**2)

  @pytest.mark.dependency(depends=["Tests_Global::test_pi_8"])
  def test_perimeter_4_poly(self, circle):
    radius = 4.2
    py_perimeter = 2.*math.pi*radius
    f_perimeter = circle.perimeter(np.float32(radius))
    assert f_perimeter == pytest.approx(py_perimeter, rel=precision)
    assert f_perimeter != pytest.approx(py_perimeter, rel=precision**2)

  @pytest.mark.dependency(depends=["Tests_Global::test_pi_8"])
  def test_perimeter_8_poly(self, circle):
    radius = 4.2
    py_perimeter = 2.*math.pi*radius
    f_perimeter = circle.perimeter(np.float64(radius))
    assert f_perimeter == pytest.approx(py_perimeter, rel=precision**2)

class Tests_abstract_type():
  def test_init_abstract(self):
    with pytest.raises(NotImplementedError):
      m_geometry.Rectangle()

  @pytest.mark.dependency()
  def test_init_child(self, square):
    assert isinstance(square, m_geometry.Square)
    assert isinstance(square, m_geometry.Rectangle)

  @pytest.mark.dependency()
  @pytest.mark.dependency(depends=["Tests_abstract_type::test_init_child"])
  def test_getter(self, square):
    assert square.length == square_size
    assert square.width == square_size

  @pytest.mark.dependency()
  def test_specific_method(self, square):
    py_perimeter = square_size*4
    f_perimeter = square.perimeter()
    assert f_perimeter == pytest.approx(py_perimeter, rel=precision)

  def test_specific_method_overload(self, square):
    assert square.is_square() == 1

  @pytest.mark.dependency(depends=["Tests_abstract_type::test_specific_method"])
  def test_multi_level_abstract(self, square):
    assert square.is_polygone() == 1

  def test_deferred_method(self, square):
    py_area = square_size**2
    f_area = square.area()
    assert f_area == pytest.approx(py_area, rel=precision)

  @pytest.mark.dependency(depends=["Tests_abstract_type::test_getter"])
  def test_setter(self, square):
    new_size = 3.6
    square.length = new_size
    square.width = new_size
    assert new_size == pytest.approx(square.length, rel=precision)
    assert new_size == pytest.approx(square.width, rel=precision)

if __name__ == '__main__':
    pytest.main()
