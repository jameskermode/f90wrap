! Adapted from http://fortranwiki.org/fortran/show/Object-oriented+programming
module ClassCircle
  implicit none
  private

  real :: pi = 3.14159

  type, public :: Circle
    private
    real :: radius
    real :: area
  contains
    procedure :: get_area => Circle_get_area
    procedure :: get_radius => Circle_get_radius
    procedure :: print_basic => Circle_print_basic, &
                 print_tagged => Circle_print_tagged
    generic :: print => print_basic, print_tagged
    final :: Circle_finalize
  end type
  interface Circle
    module procedure :: Circle_initialize
  end interface

contains

  function Circle_initialize(radius) result(this)
    type(Circle) :: this
    real, intent(in) :: radius
    this%radius = radius
    this%area = pi * radius * radius
    write(*,*) 'Initialize Circle'
    return
  end function

  subroutine Circle_finalize(this)
    type(Circle), intent(inout) :: this
    write(*,*) 'Finalizing Circle'
    return
  end subroutine

  function Circle_get_area(this) result(area)
    class(Circle), intent(in) :: this
    real :: area
    area = this%area
    return
  end function

  function Circle_get_radius(this) result(radius)
    class(Circle), intent(in) :: this
    real :: radius
    radius = this%radius
    return
  end function

  subroutine circle_print_basic(this)
    class(Circle), intent(in) :: this
    write(*,*) 'Circle: r = ', this%radius, ' area = ', this%area
    return
  end subroutine

  subroutine circle_print_tagged(this, tag)
    class(Circle), intent(in) :: this
    character(*), intent(in) :: tag
    write(*,*) 'Circle [', tag, ']: r = ', this%radius, ' area = ', this%area
    return
  end subroutine

end module

