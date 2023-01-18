
module m_circle
  implicit none
  private

  type, public :: t_square
     real :: length
  end type t_square

  type, public :: t_circle
     real :: radius
  end type t_circle

  interface is_circle
    module procedure is_circle_circle
    module procedure is_circle_square
  end interface is_circle

  interface write_array
    module procedure write_array_int_1d
    module procedure write_array_int_2d
    module procedure write_array_real
    module procedure write_array_double
  end interface write_array

  public :: is_circle
  public :: write_array
  public :: is_circle_circle
  public :: is_circle_square
  public :: write_array_int_1d

contains

  subroutine is_circle_circle(circle, output)
    type(t_circle) :: circle
    integer :: output(:)
    output(:) = 1
  end subroutine is_circle_circle

  subroutine is_circle_square(square, output)
    type(t_square) :: square
    integer :: output(:)
    output(:) = 0
  end subroutine is_circle_square

  subroutine write_array_int_1d(output)
    integer :: output(:)
    output(:) = 1
  end subroutine write_array_int_1d

  subroutine write_array_int_2d(output)
    integer :: output(:,:)
    output(:,:) = 2
  end subroutine write_array_int_2d

  subroutine write_array_real(output)
    real :: output(:)
    output(:) = 3
  end subroutine write_array_real

  subroutine write_array_double(output)
    double precision :: output(:)
    output(:) = 4
  end subroutine write_array_double

end module m_circle


