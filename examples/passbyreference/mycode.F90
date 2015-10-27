

module mymodule

implicit none

type mytype
	double precision :: val
end type mytype

contains

subroutine mysubroutine(a, b, tt)
	implicit none
	type(mytype) :: tt
	double precision, intent(in) :: a
	double precision, intent(inout) :: b

	print *,'Running mysubroutine'
	b = a*2
	tt%val = a*3

end subroutine mysubroutine

end module mymodule
