module m_array
  implicit none
  private

  type, public :: Array
     real,allocatable :: buffer(:)
     integer :: array_size
   contains
     procedure :: init => array_init
     procedure :: init_optional => array_init_optional
  end type Array

contains

  subroutine array_init(this,n)
     class(Array),target,intent(inout) :: this
     integer,              intent(in)    :: n
     this%array_size = n
     allocate(this%buffer(n))
     this%buffer = 0.0
  end subroutine array_init

  subroutine array_init_optional(this,n,optional_arg)
     class(Array),target,intent(inout) :: this
     integer,              intent(in)    :: n
     class(Array),optional,intent(in)    :: optional_arg
     this%array_size = n
     allocate(this%buffer(n))
     this%buffer = 0.0
  end subroutine array_init_optional

end module m_array
