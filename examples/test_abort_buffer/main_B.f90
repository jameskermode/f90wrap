module m_test_B
  implicit none
  private

  public :: calling_abort_B
contains

  subroutine calling_abort_B()

    call f90wrap_abort("Aborting from B")

  end subroutine

end module m_test_B
