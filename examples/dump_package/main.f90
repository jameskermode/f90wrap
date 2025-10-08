module m_array_init
  use m_array_type,only:Array
  implicit none
  public
contains
  subroutine array_init(this,n)
     type(Array),target,intent(inout) :: this
     integer,              intent(in)  :: n
     this%array_size = n
     allocate(this%buffer(n))
     this%buffer = 0.0
  end subroutine array_init
end module m_array_init
