module m_test
  implicit none
  private

  public :: t_bool_wrapper
  public :: init
  public :: free
  public :: get_scalar
  public :: get_static_array
  public :: get_dynamic_array

  type t_bool_wrapper
    logical :: scalar
    logical :: static_array(6)
    logical, allocatable :: dynamic_array(:)
  end type t_bool_wrapper

contains

  subroutine init(this, scalar, static_array, dynamic_array)
    type(t_bool_wrapper), intent(inout) :: this
    logical, intent(in) :: scalar
    logical, intent(in) :: static_array(6)
    logical, intent(in) :: dynamic_array(:)

    this%scalar = scalar
    this%static_array = static_array
    allocate(this%dynamic_array(size(dynamic_array)))
    this%dynamic_array = dynamic_array
  end subroutine init

  subroutine free(this)
    type(t_bool_wrapper), intent(inout) :: this
    if (allocated(this%dynamic_array)) then
      deallocate(this%dynamic_array)
    end if
  end subroutine free

  function get_scalar(this) result(res)
    type(t_bool_wrapper), intent(in) :: this
    logical :: res
    res = this%scalar
  end function get_scalar

  function get_static_array(this) result(res)
    type(t_bool_wrapper), intent(in) :: this
    logical :: res(6)
    res = this%static_array
  end function get_static_array

  subroutine get_dynamic_array(this, res)
    type(t_bool_wrapper), intent(in) :: this
    logical, intent(out) :: res(size(this%dynamic_array))
    res = this%dynamic_array
  end subroutine get_dynamic_array

end module m_test
