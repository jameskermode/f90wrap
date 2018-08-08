module othertype_mod

  implicit none

  type othertype
     integer :: a
  end type othertype

contains

  function constructor() result(obj)
    type(othertype) :: obj

    obj%a = 5
  end function constructor

  subroutine plus_b(obj, b, c)
    type(othertype) :: obj
    integer, intent(in) :: b
    integer, intent(out) :: c

    c = obj%a + b
  end subroutine plus_b

end module othertype_mod
