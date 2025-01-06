module m_base_poly
  implicit none
  private

  type, public, abstract :: Polygone
    contains
      procedure :: is_polygone => polygone_is_polygone
  end type Polygone
contains
  function polygone_is_polygone(this) result(is_polygone)
    class(Polygone), intent(in) :: this
    integer :: is_polygone
    is_polygone = 1
  end function polygone_is_polygone
end module m_base_poly


