module m_array_type
  implicit none
  private

  type, public :: Array
     real,allocatable :: buffer(:)
     integer :: array_size
  end type Array


end module m_array_type
