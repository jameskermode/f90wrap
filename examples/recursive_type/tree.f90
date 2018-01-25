module tree
  implicit none
  type node
     type(node), pointer :: left=>null()
     type(node), pointer :: right=>null()

  end type node

contains

  subroutine treeAllocate(root)
    implicit none

    type(node) :: root
    allocate(root%left)
    allocate(root%right)

  end subroutine treeAllocate

 subroutine treedeallocate(root)
    implicit none

    type(node) :: root
    deallocate(root%left, root%right)

  end subroutine treedeallocate


end module tree
