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

subroutine routine_member_procedures2(in1, in2, out1, out2)
  ! Test member subroutine and function
  implicit none
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2
  integer :: localvar 
  
  localvar = in2
  call member_procedure(in1, out1)
  out2 = member_function(out1)
  localvar = out2
  call member_procedure2(localvar, out1) 
  out2 = member_function2(out1)
  localvar = out2
  call member_procedure3(localvar, out1)
  out2 = member_function3(out1)
contains
  subroutine member_procedure(in1, out1)
    ! This member procedure shadows some variables and uses
    ! a variable from the parent scope
    implicit none
    integer, intent(in) :: in1
    integer, intent(out) :: out1
    
    out1 = 5 * in1 + localvar
  end subroutine member_procedure
  subroutine member_procedure2(in1, out1)
    ! This member procedure shadows some variables and uses
    ! a variable from the parent scope
    implicit none
    integer, intent(in) :: in1
    integer, intent(out) :: out1
    
    out1 = in1 - 10
  end subroutine member_procedure2
  subroutine member_procedure3(in1, out1)
    ! This member procedure shadows some variables and uses
    ! a variable from the parent scope
    implicit none
    integer, intent(in) :: in1
    integer, intent(out) :: out1
    
    out1 = in1 + 2
  end subroutine member_procedure3
  function member_function(a) result(b)
    implicit none
    integer, intent(in) :: a
    integer :: b
    
    b = 3 * a + 2
  end function member_function
  function member_function2(a) result(b)
    implicit none
    integer, intent(in) :: a
    integer :: b
    
    b = 2 * a
  end function member_function2
  function member_function3(a) result(b)
    implicit none
    integer, intent(in) :: a
    integer :: b
    
    b = 3 * a
  end function member_function3
end subroutine routine_member_procedures2

function function_member_procedures(in1, in2, out1, out2) result(out3)
  ! Test member subroutine and function
  implicit none
  integer, intent(in) :: in1, in2
  integer, intent(out) :: out1, out2
  integer :: out3
  
  out1 = member_function(in1)
  out2 = member_function2(in2)
  out3 = out1 + out2
contains
  function member_function(a) result(b)
    implicit none
    integer, intent(in) :: a
    integer :: b
    
    b = 3 * a + 2
  end function member_function
  function member_function2(a) result(b)
    implicit none
    integer, intent(in) :: a
    integer :: b
    
    b = 2 * a
  end function member_function2
end function function_member_procedures
