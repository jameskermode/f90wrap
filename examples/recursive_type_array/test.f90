module mod_recursive_type_array
  implicit none
  type t_node
    type(t_node),dimension(:),allocatable :: node
  end type t_node
  
contains

  subroutine allocate_node(root,N_node)
    type(t_node) :: root
    integer :: N_node

    allocate(root%node(N_node))

  end subroutine allocate_node

  subroutine deallocate_node(root)
    type(t_node) :: root

    if (allocated(root%node)) deallocate(root%node)

  end subroutine
end module mod_recursive_type_array
