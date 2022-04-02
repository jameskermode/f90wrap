module my_module
implicit none

type mytype
   integer :: n=0,m=0
   real(8), allocatable :: y(:,:)
end type mytype
  
contains

subroutine allocit(x,n,m)
  implicit none
  integer, intent(in) :: n,m
  type(mytype), intent(inout) :: x
  integer i,j

  write(6,*) 'allocit> n,m=',n,m

  x%n = n; x%m = m
  allocate(x%y(n,m))

  do i = 1, n 
   do j = 1, m
    x%y(i,j) = real(i+j)/real(n+m)
   end do
  end do

end subroutine allocit

end module my_module
