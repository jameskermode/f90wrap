
module m_type_test
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
    module procedure write_array_int32_0d
    module procedure write_array_int64_0d
    module procedure write_array_real32_0d
    module procedure write_array_real64_0d
    module procedure write_array_int_1d
    module procedure write_array_int_2d
    module procedure write_array_real
    module procedure write_array_double
    module procedure write_array_bool
  end interface write_array

  interface optional_scalar
    module procedure optional_scalar_real
    module procedure optional_scalar_int
  end interface optional_scalar

  interface in_scalar
    module procedure in_scalar_int8
    module procedure in_scalar_int16
    module procedure in_scalar_int32
    module procedure in_scalar_int64
    module procedure in_scalar_real32
    module procedure in_scalar_real64
    module procedure in_array_int64
    module procedure in_array_real64
  end interface in_scalar

  public :: is_circle
  public :: write_array
  public :: is_circle_circle
  public :: is_circle_square
  public :: write_array_int_1d
  public :: optional_scalar

  public :: write_array_int64_0d
  public :: write_array_real64_0d
  public :: write_array_real
  public :: write_array_double

  public :: in_scalar
  public :: in_scalar_int8
  public :: in_scalar_int16
  public :: in_scalar_int32
  public :: in_scalar_int64
  public :: in_scalar_real32
  public :: in_scalar_real64
  public :: in_array_int64
  public :: in_array_real64

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

  subroutine write_array_int32_0d(output)
    integer(kind=4),intent(inout) :: output
    output = 10
  end subroutine write_array_int32_0d

  subroutine write_array_int64_0d(output)
    integer(kind=8),intent(inout) :: output
    output = 11
  end subroutine write_array_int64_0d

  subroutine write_array_real32_0d(output)
    real(kind=4),intent(inout) :: output
    output = 12
  end subroutine write_array_real32_0d

  subroutine write_array_real64_0d(output)
    real(kind=8),intent(inout) :: output
    output = 13
  end subroutine write_array_real64_0d

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

  subroutine write_array_bool(output)
    logical :: output(:)
    output(:) = .true.
  end subroutine write_array_bool

  subroutine optional_scalar_real(output, opt_output)
    real, intent(inout)         :: output(:)
    real, intent(out),optional  :: opt_output
    output(:) = 10
    if (present(opt_output)) then
      opt_output = 20
    endif
  end subroutine

  subroutine optional_scalar_int(output, opt_output)
    integer, intent(inout)         :: output(:)
    integer, intent(out),optional  :: opt_output
    output(:) = 15
    if (present(opt_output)) then
      opt_output = 25
    endif
  end subroutine

  function in_scalar_int8(input) result(output)
    integer(kind=1),intent(in) :: input
    integer(kind=4)            :: output
    output = 108
  end function in_scalar_int8

  function in_scalar_int16(input) result(output)
    integer(kind=2),intent(in) :: input
    integer(kind=4)            :: output
    output = 116
  end function in_scalar_int16

  function in_scalar_int32(input) result(output)
    integer(kind=4),intent(in) :: input
    integer(kind=4)            :: output
    output = 132
  end function in_scalar_int32

  function in_scalar_int64(input) result(output)
    integer(kind=8),intent(in) :: input
    integer(kind=4)            :: output
    output = 164
  end function in_scalar_int64

  function in_scalar_real32(input) result(output)
    real(kind=4),intent(in)    :: input
    integer(kind=4)            :: output
    output = 232
  end function in_scalar_real32

  function in_scalar_real64(input) result(output)
    real(kind=8),intent(in)    :: input
    integer(kind=4)            :: output
    output = 264
  end function in_scalar_real64

  function in_array_int64(input) result(output)
    integer(kind=8),intent(in) :: input(:)
    integer(kind=4)            :: output
    output = 364
  end function in_array_int64

  function in_array_real64(input) result(output)
    real(kind=8),intent(in)    :: input(:)
    integer(kind=4)            :: output
    output = 464
  end function in_array_real64

end module m_type_test
