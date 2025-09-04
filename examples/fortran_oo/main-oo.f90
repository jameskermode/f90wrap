module m_geometry
  use iso_c_binding
  use m_base_poly, only : Polygone
  implicit none
  private
  real(kind=8), parameter :: pi = atan(1.d0)*4.0d0  ! Class-wide private constant

  type, public, abstract, extends(Polygone) :: Rectangle
      real(kind=8) :: length
      real(kind=8) :: width
    contains
      procedure :: perimeter => rectangle_perimeter
      procedure :: is_square => rectangle_is_square
      procedure(abstract_area), deferred :: area
  end type Rectangle

  type, public, extends(Rectangle) :: Square
    contains
      procedure :: init => square_init
      procedure :: is_square => square_is_square
      procedure :: area => square_area
      procedure :: is_equal => square_is_equal
      procedure :: copy => square_copy
      procedure :: create_diamond => square_create_diamond
      generic   :: assignment(=) => copy
  end type Square

  type, public, extends(Polygone) :: Diamond
      real(kind=8) :: length
      real(kind=8) :: width
    contains
      procedure :: init => diamond_init
      procedure :: info => diamond_info
      procedure :: copy => diamond_copy
      generic   :: assignment(=) => copy
  end type Diamond

  abstract interface
    function abstract_area(this) result(area)
      import Rectangle
      real(kind=8) :: area
      class(Rectangle), intent(in)  :: this
    end function abstract_area
  end interface

  interface Square
    module procedure :: construct_square
  end interface Square

  type,public :: List_square
     type(square), allocatable :: alloc_type(:)
     type(square), pointer     :: ptr_type(:)
     class(square),allocatable :: alloc_class(:)
     class(square),pointer     :: ptr_class(:)
     class(square),pointer     :: scalar_class
     type(square)              :: scalar_type
     integer :: n
   contains
     procedure :: init => list_square_init
  end type List_square

  type, public :: Circle
     real(kind=8) :: radius
   contains
     procedure :: area => circle_area
     procedure :: print => circle_print
     procedure :: obj_name => circle_obj_name
     procedure :: copy => circle_copy
     procedure :: init => circle_init
     procedure :: private_method => circle_private
     procedure :: perimeter_4 => circle_perimeter_4
     procedure :: perimeter_8 => circle_perimeter_8
     generic   :: perimeter => perimeter_8, perimeter_4
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

  type,public :: List_circle
     type(Circle), allocatable :: alloc_type(:)
     type(Circle), pointer     :: ptr_type(:)
     class(Circle),allocatable :: alloc_class(:)
     class(Circle),pointer     :: ptr_class(:)
     class(Circle),pointer     :: scalar_class
     type(Circle)              :: scalar_type
     integer :: n
   contains
     procedure :: init => list_circle_init
  end type List_circle

  type, public :: Array
     real,allocatable :: buf(:)
     real,pointer     :: values(:) => null()
   contains
     procedure :: init => array_init
  end type Array

  type, public, extends(Array) :: Array_3d
     real,pointer :: values_3d(:,:,:) => null()
   contains
     procedure :: init_3d => array_3d_init
  end type Array_3d

  public:: pi
  public:: circle_area,circle_print,circle_obj_name
  public:: ball_area,ball_volume
  public:: circle_copy,circle_free

  public:: get_circle_radius,get_ball_radius
contains

  subroutine array_init(this,n)
     class(Array),target,intent(inout) :: this
     integer,              intent(in)    :: n
     allocate(this%buf(n))
     this%values => this%buf
  end subroutine array_init

  subroutine array_3d_init(this,n1,n2,n3)
     class(Array_3d),target,intent(inout) :: this
     integer,                 intent(in)    :: n1,n2,n3

     type(c_ptr) :: cptr

     call this%Array%init(n1*n2*n3)

     cptr=c_loc(this%values)
     call c_f_pointer(cptr,this%values_3d,shape=(/n1,n2,n3/))

  end subroutine array_3d_init

  function construct_square(length)
    type(Square)     :: construct_square
    real, intent(in) :: length
    construct_square%length = length
    construct_square%width = length
  end function construct_square

  subroutine list_square_init(this,n)
    class(List_square),intent(inout) :: this
    integer,          intent(in) :: n
    this%n = n
    allocate(square :: this%alloc_class(n))
    allocate(square :: this%ptr_class(n))
    allocate(this%alloc_type(n))
    allocate(this%ptr_type(n))
    allocate(square :: this%scalar_class)
  end subroutine list_square_init

  subroutine list_circle_init(this,n)
    class(List_circle),intent(inout) :: this
    integer,          intent(in) :: n
    this%n = n
    allocate(Circle :: this%alloc_class(n))
    allocate(Circle :: this%ptr_class(n))
    allocate(this%alloc_type(n))
    allocate(this%ptr_type(n))
    allocate(Circle :: this%scalar_class)
  end subroutine list_circle_init

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
    real(kind=8) :: radius
    radius = my_circle%radius
  end function get_circle_radius

  function get_ball_radius(my_ball) result(radius)
    class(Ball), intent(in) :: my_ball
    real(kind=8) :: radius
    radius = my_ball%radius
  end function get_ball_radius

  function circle_area(this) result(area)
    class(Circle), intent(in) :: this
    real(kind=8) :: area
    area = pi * this%radius**2
  end function circle_area

  subroutine circle_print(this)
    class(Circle), intent(in) :: this
    real(kind=8) :: area
    area = this%area()  ! Call the type-bound function
  end subroutine circle_print

  subroutine circle_obj_name(obj)
    class(Circle), intent(in) :: obj
    real(kind=8) :: area
    area = obj%area()  ! Call the type-bound function
  end subroutine circle_obj_name

  subroutine circle_copy(this, from)
    class(Circle), intent(inout) :: this
    class(Circle), intent(in) :: from
    this%radius = from%radius
  end subroutine circle_copy

  subroutine circle_init(this, radius)
    class(Circle), intent(inout) :: this
    real,          intent(in) :: radius
    this%radius = radius
  end subroutine circle_init

  subroutine circle_private(this)
    class(Circle), intent(in) :: this
  end subroutine circle_private

  subroutine circle_free(this)
    type(Circle), intent(inout) :: this
  end subroutine circle_free

  function ball_area(this) result(area)
    class(Ball), intent(in) :: this
    real(kind=8) :: area
    area = 4.0d0 * pi * this%radius**2
  end function ball_area

  function ball_volume(this) result(volume)
    class(Ball), intent(in) :: this
    real(kind=8) :: volume
    volume = 4.0d0/3.0d0 * pi * this%radius**3
  end function ball_volume

  subroutine ball_private(this)
    class(Ball), intent(in) :: this
  end subroutine ball_private

  function circle_perimeter_4(this, radius) result(perimeter)
    class(Circle), intent(in) :: this
    real(kind=4), intent(in) :: radius
    real(kind=4) :: perimeter
    perimeter = 2.0 * pi * radius
  end function circle_perimeter_4

  function circle_perimeter_8(this, radius) result(perimeter)
    class(Circle), intent(in) :: this
    real(kind=8), intent(in) :: radius
    real(kind=8) :: perimeter
    perimeter = 2.0d0 * pi * radius
  end function circle_perimeter_8

  function rectangle_perimeter(this) result(perimeter)
    class(Rectangle), intent(in) :: this
    real(kind=8) :: perimeter
    perimeter = 2*this%length + 2*this%width
  end function rectangle_perimeter

  function square_area(this) result(area)
    class(Square), intent(in) :: this
    real(kind=8) :: area
    area = this%length * this%length
  end function square_area

  function rectangle_is_square(this) result(is_square)
    class(Rectangle), intent(in) :: this
    integer :: is_square
    is_square = 0
  end function rectangle_is_square

  subroutine square_copy(this, from)
    class(Square), intent(inout) :: this
    class(Square), intent(in) :: from
    this%length = from%length
    this%width = from%width
  end subroutine square_copy

  subroutine square_init(this, length)
    class(Square), intent(inout) :: this
    real,          intent(in) :: length
    this%length = length
    this%width = length
  end subroutine square_init

  function square_is_square(this) result(is_square)
    class(Square), intent(in) :: this
    integer :: is_square
    is_square = 1
  end function square_is_square

  function square_is_equal(this,other) result(is_equal)
    class(Square),          intent(in) :: this
    class(Rectangle),target,intent(in) :: other
    integer :: is_equal

    is_equal = 0

    if (other%is_square() .eq. 1) then
      if (other%length == this%length) then
         is_equal = 1
      endif
    endif
  end function square_is_equal

  subroutine diamond_init(this,width,length)
    class(Diamond), intent(inout) :: this
    real(kind=8),   intent(in)    :: width,length
    this%width=width
    this%length=length
  end subroutine diamond_init

  subroutine diamond_copy(this,other)
    class(Diamond), intent(inout) :: this
    type(Diamond),  intent(in)    :: other
    call this%init(other%width,other%length)
  end subroutine diamond_copy

  subroutine diamond_info(this)
    class(Diamond), intent(in) :: this
    print *,'Diamong width =',this%width
    print *,'Diamong length =',this%length
  end subroutine diamond_info

  function square_create_diamond(this)
    class(Square), intent(in) :: this
    type(diamond)             :: square_create_diamond

    call square_create_diamond%init(this%width,this%width)
  end function square_create_diamond


end module m_geometry
