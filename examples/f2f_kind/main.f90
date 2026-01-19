module m_kind_test
  implicit none
  private

  public :: kind_test

contains

  subroutine kind_test(an_int, a_real)

    implicit none
    integer(kind=PREPROCESSOR_DEFINED_INT),  intent(inout) :: an_int
    real(kind=PREPROCESSOR_DEFINED_REAL),    intent(inout) :: a_real

    an_int = 1
    a_real = 1.0

  end subroutine

end module m_kind_test
