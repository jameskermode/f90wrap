module m_test_A
  implicit none
  private

  public :: calling_abort_A
contains

  subroutine calling_abort_A()

    call f90wrap_abort("Aborting from A")

  end subroutine

end module m_test_A
