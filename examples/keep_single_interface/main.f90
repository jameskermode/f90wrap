module m_test
  implicit none
  private

  interface an_interface
    module procedure a_subroutine
  end interface an_interface

  public :: an_interface,a_subroutine

contains

  subroutine a_subroutine(input, output)
    real, intent(in)  :: input
    real, intent(out)  :: output
    output = input * 2.0
  end subroutine

end module m_test
