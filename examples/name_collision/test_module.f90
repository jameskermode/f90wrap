! Test case for name collision between subroutine name and argument name
! This reproduces the enable_timing bug found in QUIP

module test_module
  implicit none

contains

  ! This subroutine exists in the module
  subroutine enable_timing()
    print *, "Timing enabled"
  end subroutine enable_timing

  ! This subroutine has an argument with the same name as the above subroutine
  ! When f90wrap generates "use test_module", both symbols are imported
  ! causing a name collision
  subroutine system_init(enable_timing, verbosity)
    logical, intent(in) :: enable_timing
    integer, intent(in), optional :: verbosity

    if (enable_timing) then
      print *, "System initialized with timing"
    else
      print *, "System initialized without timing"
    end if

    if (present(verbosity)) then
      print *, "Verbosity level:", verbosity
    end if
  end subroutine system_init

end module test_module
