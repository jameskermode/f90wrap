subroutine routine_member_procedures(in1, in2, out1, out2)
  ! Test member subroutine and function
  implicit none
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2
  integer :: localvar 
  
  localvar = in2
  call member_procedure(in1, out1)
  out2 = member_function(out1)
contains
  subroutine member_procedure(in1, out1)
    ! This member procedure shadows some variables and uses
    ! a variable from the parent scope
    implicit none
    integer, intent(in) :: in1
    integer, intent(out) :: out1
    
    out1 = 5 * in1 + localvar
  end subroutine member_procedure
  function member_function(a) result(b)
    implicit none
    integer, intent(in) :: a
    integer :: b
    
    b = 3 * a + 2
  end function member_function
end subroutine routine_member_procedures
