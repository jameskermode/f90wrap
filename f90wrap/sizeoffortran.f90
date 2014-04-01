subroutine sizeof_fortran_t(a)

  type ptr_type
     type(ptr_type), pointer :: p => NULL()
  end type ptr_type
  type(ptr_type) :: ptr
  integer, allocatable, dimension(:) :: ptr_int
  integer, intent(out) :: a

  a =  size(transfer(ptr, ptr_int))

end subroutine sizeof_fortran_t
