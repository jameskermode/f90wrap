program f90wrap_test
use highest_level
implicit none

! local variables
integer, parameter :: N=5
real :: rcond,determ
real :: atemp(N,N)
integer :: i 

type(outer) :: size_data

do i=1,N
    atemp(i,i) = real(i)
end do 
write(*,*) 'hello',kind(atemp(1,1))
call invert(atemp,N,N,rcond,determ)
write(*,*) atemp
write(*,*) 'from pyinv_mod'
write(*,*) size_data%size%test_double, kind(size_data%size%test_double)

!associate pointer in py_invisible_mod
!if(associated(size_data)) nullify(size_data); size_data => data
!if(associated(size_data)) nullify(size_data)

end program