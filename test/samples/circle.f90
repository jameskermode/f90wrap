module ClassCircle
  implicit none

  real, private :: pi = 3.14159

  type :: Circle
    private
    character(:), allocatable :: name
    real :: radius
    real :: area
  contains
    procedure :: get_area => Circle_get_area
    procedure :: get_radius => Circle_get_radius
    procedure, private, non_overridable :: &
      print_basic => Circle_print_basic,   &
      print_tagged => Circle_print_tagged
    generic :: print => print_basic, print_tagged
    final :: Circle_finalize
    final :: Circle_array_finalize
  end type
  interface Circle
    module procedure :: Circle_named_initialize
    module procedure :: Circle_unnamed_initialize
  end interface

contains

  function Circle_named_initialize(name, radius) result(this)
    type(Circle) :: this
    character(*), intent(in) :: name
    real, intent(in) :: radius
    this%name = name
    this%radius = radius
    this%area = pi * radius * radius
    write(*,*) 'Circle(', this%name, '): Construct w/ name'
    return
  end function

  function Circle_unnamed_initialize(radius) result(this)
    type(Circle) :: this
    real, intent(in) :: radius
    this%name = 'unnamed'
    this%radius = radius
    this%area = pi * radius * radius
    write(*,*) 'Circle(unnamed): Construct w/o name'
    return
  end function

  subroutine Circle_finalize(this)
    type(Circle), intent(inout) :: this
    write(*,*) 'Circle(', this%name, '): Destruct'
    return
  end subroutine

  subroutine Circle_array_finalize(this_arr)
    type(Circle), intent(inout) :: this_arr(:)
    integer :: i
    do i = 1, size(this_arr)
      write(*,*) 'Circle(', this_arr(i)%name, '): Destruct'
    end do
    return
  end subroutine

  function Circle_get_area(this) result(area)
    class(Circle), intent(in) :: this
    real :: area
    area = this%area
    write(*,*) 'Circle(', this%name, '): Get Area'
    return
  end function

  function Circle_get_radius(this) result(radius)
    class(Circle), intent(in) :: this
    real :: radius
    radius = this%radius
    write(*,*) 'Circle(', this%name, '): Get Radius'
    return
  end function

  subroutine circle_print_basic(this)
    class(Circle), intent(in) :: this
    write(*,*) 'Circle(', this%name, '): r = ', this%radius, ' area = ', this%area
    return
  end subroutine

  subroutine circle_print_tagged(this, tag)
    class(Circle), intent(in) :: this
    character(*), intent(in) :: tag
    write(*,*) 'Circle(', this%name, ') [', tag, ']: r = ', this%radius, ' area = ', this%area
    return
  end subroutine

end module

