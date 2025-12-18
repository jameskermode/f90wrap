module alloc_mod
  implicit none

  ! Module-level allocatable array
  real, allocatable, dimension(:,:) :: data_array

contains

  subroutine allocate_array(n, m)
    integer, intent(in) :: n, m

    if (allocated(data_array)) deallocate(data_array)
    allocate(data_array(n, m))
    data_array = 0.0
  end subroutine allocate_array

  subroutine fill_array(val)
    real, intent(in) :: val

    if (allocated(data_array)) then
      data_array = val
    end if
  end subroutine fill_array

  subroutine reallocate_array(n, m)
    ! Reallocate with different dimensions
    integer, intent(in) :: n, m

    if (allocated(data_array)) deallocate(data_array)
    allocate(data_array(n, m))
    data_array = 0.0
  end subroutine reallocate_array

end module alloc_mod
