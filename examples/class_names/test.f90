module module_snake_mod
  type ceci_ne_pas_un_chameau
     integer :: y
  end type ceci_ne_pas_un_chameau
  
  type array_type
     type(ceci_ne_pas_un_chameau) :: x(2)
  end type array_type

  type(array_type), dimension(10), target :: xarr
  
contains
  subroutine recup_point(x)
    type(array_type) :: x
    return
  end subroutine recup_point
end module module_snake_mod
