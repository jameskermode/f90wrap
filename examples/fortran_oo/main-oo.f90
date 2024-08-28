module m_geometry
  use m_base_poly, only : Polygone
  implicit none
  private
  real(kind=8) :: pi = 3.1415926535897931d0 ! Class-wide private constant

  type, public, abstract, extends(Polygone) :: Rectangle
      real :: length
      real :: width
    contains
      procedure :: perimeter => rectangle_perimeter
      procedure :: is_square => rectangle_is_square
      procedure(abstract_area), deferred :: area
  end type Rectangle

  type, public, extends(Rectangle) :: Square
    contains
      procedure :: is_square => square_is_square
      procedure :: area => square_area
  end type Square

  abstract interface
    function abstract_area(this)
      import Rectangle
      class(Rectangle), intent(in)  :: this
    end function abstract_area
  end interface

  interface Square
    module procedure :: construct_square
  end interface Square

  type, public :: Circle
     real :: radius
   contains
     procedure :: area => circle_area
     procedure :: print => circle_print
     procedure :: obj_name => circle_obj_name
     procedure :: copy => circle_copy
     procedure :: private_method => circle_private
     procedure :: perimeter_4 => circle_perimeter_4
     procedure :: perimeter_8 => circle_perimeter_8
     generic :: perimeter => perimeter_8, perimeter_4
     final     :: circle_free
  end type Circle

  type, public, extends(Circle) :: Ball
   contains
     procedure :: volume => ball_volume
     procedure :: area => ball_area
     procedure :: private_method => ball_private
  end type Ball

  interface Circle
    module procedure :: construct_circle
  end interface Circle

  interface Ball
    module procedure :: construct_ball
  end interface Ball

  public:: pi
  public:: circle_area,circle_print,circle_obj_name
  public:: ball_area,ball_volume
  public:: circle_copy,circle_free

  public:: get_circle_radius,get_ball_radius
contains

  function construct_square(length)
    type(Square) :: construct_square
    real, intent(in) :: length
    construct_square%length = length
    construct_square%width = length
  end function construct_square

  function construct_circle(rc,rb)
    type(Circle) :: construct_circle
    real, intent(in) :: rc,rb
    construct_circle%radius = rc
  end function construct_circle

  function construct_ball(rc,rb)
    type(Ball) :: construct_ball
    real, intent(in) :: rc,rb
    construct_ball%radius = rb
  end function construct_ball

  function get_circle_radius(my_circle) result(radius)
    class(Circle), intent(in) :: my_circle
    real :: radius
    radius = my_circle%radius
  end function get_circle_radius

  function get_ball_radius(my_ball) result(radius)
    class(Ball), intent(in) :: my_ball
    real :: radius
    radius = my_ball%radius
  end function get_ball_radius

  function circle_area(this) result(area)
    class(Circle), intent(in) :: this
    real :: area
    area = pi * this%radius**2
  end function circle_area

  subroutine circle_print(this)
    class(Circle), intent(in) :: this
    real :: area
    area = this%area()  ! Call the type-bound function
  end subroutine circle_print

  subroutine circle_obj_name(obj)
    class(Circle), intent(in) :: obj
    real :: area
    area = obj%area()  ! Call the type-bound function
  end subroutine circle_obj_name

  subroutine circle_copy(this, from)
    class(Circle), intent(inout) :: this
    class(Circle), intent(in) :: from
    this%radius = from%radius
  end subroutine circle_copy

  subroutine circle_private(this)
    class(Circle), intent(in) :: this
  end subroutine circle_private

  subroutine circle_free(this)
    type(Circle), intent(inout) :: this
  end subroutine circle_free

  function ball_area(this) result(area)
    class(Ball), intent(in) :: this
    real :: area
    area = 4. * pi * this%radius**2
  end function ball_area

  function ball_volume(this) result(volume)
    class(Ball), intent(in) :: this
    real :: volume
    volume = 4./3. * pi * this%radius**3
  end function ball_volume

  subroutine ball_private(this)
    class(Ball), intent(in) :: this
  end subroutine ball_private

  function circle_perimeter_4(this, radius) result(perimeter)
    class(Circle), intent(in) :: this
    real(kind=4), intent(in) :: radius
    real(kind=4) :: perimeter
    perimeter = 2. * pi * radius
  end function circle_perimeter_4

  function circle_perimeter_8(this, radius) result(perimeter)
    class(Circle), intent(in) :: this
    real(kind=8), intent(in) :: radius
    real(kind=8) :: perimeter
    perimeter = 2. * pi * radius
  end function circle_perimeter_8

  function rectangle_perimeter(this) result(perimeter)
    class(Rectangle), intent(in) :: this
    real :: perimeter
    perimeter = 2*this%length + 2*this%width
  end function rectangle_perimeter

  function square_area(this) result(area)
    class(Square), intent(in) :: this
    real :: area
    area = this%length * this%length
  end function square_area

  function rectangle_is_square(this) result(is_square)
    class(Rectangle), intent(in) :: this
    integer :: is_square
    is_square = 0
  end function rectangle_is_square

  function square_is_square(this) result(is_square)
    class(Square), intent(in) :: this
    integer :: is_square
    is_square = 1
  end function square_is_square

end module m_geometry


