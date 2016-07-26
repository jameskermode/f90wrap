module module_calcul
type type_ptmes
integer :: y
end type type_ptmes
type array_type
type(type_ptmes) :: x(2)
end type array_type
contains
subroutine recup_point(x)
type(array_type) :: x
return
end subroutine recup_point
end module module_calcul