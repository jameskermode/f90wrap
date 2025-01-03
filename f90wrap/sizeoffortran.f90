subroutine sizeof_fortran_t(size_out)

  type ptr_type
     type(ptr_type), pointer :: p => NULL()
  end type ptr_type
  type(ptr_type) :: ptr_t
  integer, allocatable, dimension(:) :: ptr_int_t

  type ptr_class
     class(ptr_class), pointer :: p => NULL()
  end type ptr_class
  type(ptr_class) :: ptr_c
  integer, allocatable, dimension(:) :: ptr_int_c

  integer, intent(out) :: size_out

  ptr_t_size =  size(transfer(ptr_t, ptr_int_t))
  ptr_c_size =  size(transfer(ptr_c, ptr_int_c))
  size_out = max(ptr_t_size, ptr_c_size)

end subroutine sizeof_fortran_t
