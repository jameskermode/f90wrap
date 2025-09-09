
module m_fortran_module
  use m_base_type, only: t_base_type
  use m_inheritance, only: t_inheritance
  use m_composition, only: t_composition
  implicit none
  private

  public :: a_subroutine, b_subroutine, c_subroutine

contains

  subroutine a_subroutine(input)
    type(t_base_type),intent(inout) :: input
    input%real_number=1.0
  end subroutine

  subroutine b_subroutine(input)
    type(t_inheritance),intent(inout) :: input
    input%real_number=1.0
    input%integer_number=2.0
  end subroutine

  subroutine c_subroutine(input)
    type(t_composition),intent(inout) :: input
    input%member%real_number=1.0
  end subroutine

end module m_fortran_module
