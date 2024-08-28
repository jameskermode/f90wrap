program main

end program

module m_test

  implicit none
  public

contains

  function to_be_ignored_1() result(ptr)
    real,pointer              :: ptr(:,:)
    ptr => null()
  end function to_be_ignored_1

  function to_be_ignored_2() result(ptr)
    real,pointer,contiguous   :: ptr(:,:)
    ptr => null()
  end function to_be_ignored_2

  function not_to_be_ignored() result(out_int)
    integer                   :: out_int
    out_int = 1
  end function not_to_be_ignored

end module m_test
