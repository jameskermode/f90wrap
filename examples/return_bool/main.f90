module m_test
  implicit none
  private

  public :: return_logical

contains

  logical function return_logical(flag)
    logical, intent(in)  :: flag

    return_logical = flag

  end function return_logical

end module m_test
