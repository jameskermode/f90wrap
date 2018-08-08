module mytype_mod

  implicit none

  type mytype
     integer :: a
  end type mytype

contains

  function constructor() result(obj)
    type(mytype) :: obj

    obj%a = 2
  end function constructor

  subroutine plus_b(obj, b, c)
    type(mytype) :: obj
    integer, intent(in) :: b
    integer, intent(out) :: c

    c = obj%a + b
  end subroutine plus_b

end module mytype_mod
