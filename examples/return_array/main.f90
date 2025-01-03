module m_test
  implicit none
  private

  type, public :: t_array_wrapper
     integer :: a_size
     real,allocatable :: a_data(:)
  end type t_array_wrapper

  type, public :: t_array_2d_wrapper
     integer :: a_size_x, a_size_y
     real,allocatable :: a_data(:,:)
  end type t_array_2d_wrapper

  type, public :: t_array_double_wrapper
     type(t_array_wrapper)  array_wrapper
  end type t_array_double_wrapper

  type, public :: t_value
     real :: value
  end type t_value

  type, public :: t_size_2d
     integer :: x, y
  end type t_size_2d

  public :: array_init, array_free
  public :: array_wrapper_init
  public :: array_2d_init
  public :: return_scalar
  public :: return_hard_coded_1d
  public :: return_hard_coded_2d
  public :: return_array_member
  public :: return_array_member_2d
  public :: return_array_member_wrapper
  public :: return_array_input
  public :: return_array_input_2d
  public :: return_array_size
  public :: return_array_size_2d_in
  public :: return_array_size_2d_out
  public :: return_derived_type_value

contains

  subroutine array_init(in_array, in_size)
    type(t_array_wrapper), intent(inout)  :: in_array
    integer, intent(in)                   :: in_size

    in_array%a_size = in_size
    allocate(in_array%a_data(in_array%a_size))
    in_array%a_data = 1
  end subroutine array_init

  subroutine array_2d_init(in_array, in_size_x, in_size_y)
    type(t_array_2d_wrapper), intent(inout)   :: in_array
    integer, intent(in)                       :: in_size_x, in_size_y

    in_array%a_size_x = in_size_x
    in_array%a_size_y = in_size_y
    allocate(in_array%a_data(in_array%a_size_x, in_array%a_size_y))
    in_array%a_data = 2
  end subroutine array_2d_init

  subroutine array_wrapper_init(in_wrapper, in_size)
    type(t_array_double_wrapper), intent(inout)  :: in_wrapper
    integer, intent(in)                   :: in_size

    in_wrapper%array_wrapper%a_size = in_size
    allocate(in_wrapper%array_wrapper%a_data(in_wrapper%array_wrapper%a_size))
    in_wrapper%array_wrapper%a_data = 2
  end subroutine array_wrapper_init

  subroutine array_free(in_array)
    type(t_array_wrapper), intent(inout)  :: in_array

    in_array%a_size = 0
    deallocate(in_array%a_data)
  end subroutine array_free

  function return_scalar(in_array)
    type(t_array_wrapper), intent(inout)  :: in_array
    real                                  :: return_scalar

    return_scalar=in_array%a_data(1)
  end function return_scalar

  function return_hard_coded_1d() result(retval)
    real                                  :: retval(10)

    retval=2
  end function return_hard_coded_1d

  function return_hard_coded_2d() result(retval)
    real                                  :: retval(5,6)

    retval=3
  end function return_hard_coded_2d

  function return_array_member(in_array) result(retval)
    type(t_array_wrapper), intent(inout)  :: in_array
    real                                  :: retval(in_array%a_size)

    retval=in_array%a_data
  end function return_array_member

  function return_array_member_2d(in_array) result(retval)
    type(t_array_2d_wrapper), intent(inout)   :: in_array
    real                                      :: retval(in_array%a_size_x, in_array%a_size_y)

    retval=in_array%a_data
  end function return_array_member_2d

  function return_array_member_wrapper(in_wrapper) result(retval)
    type(t_array_double_wrapper), intent(inout)  :: in_wrapper
    real                                  :: retval(in_wrapper%array_wrapper%a_size)

    retval=in_wrapper%array_wrapper%a_data
  end function return_array_member_wrapper

  function return_array_input(in_len) result(retval)
    integer,   intent(in)  :: in_len
    real :: retval(in_len)

    retval = 1
  end function return_array_input

  function return_array_input_2d(in_len_x, in_len_y) result(retval)
    integer,   intent(in)  :: in_len_x,in_len_y
    real :: retval(in_len_x, in_len_y)

    retval = 2
  end function return_array_input_2d

  function return_array_size(in_array) result(retval)
    real,   intent(in)  :: in_array(:)
    real :: retval(size(in_array))

    retval = 1
  end function return_array_size

  function return_array_size_2d_in(in_array) result(retval)
    real,   intent(in)  :: in_array(:,:)
    real :: retval(size(in_array,2))

    retval = 1
  end function return_array_size_2d_in

  function return_array_size_2d_out(in_array_1, in_array_2) result(retval)
    real,   intent(in)  :: in_array_1(:,:)
    real,   intent(in)  :: in_array_2(:,:)
    real :: retval(size(in_array_1,1), size(in_array_2,2))

    retval = 2
  end function return_array_size_2d_out

  function return_derived_type_value(this,size_2d) result(output)
    type(t_value),    intent(in)  :: this
    type(t_size_2d),  intent(in)  :: size_2d
    real                          :: output(size_2d%x,size_2d%y)

    output = this%value
  end function return_derived_type_value

end module m_test
