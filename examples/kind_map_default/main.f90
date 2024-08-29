program main

end program

module m_test

  implicit none
  public

contains

  function test_real(in_real) result(out_int)
    real          :: in_real
    integer       :: out_int
    out_int = 1
  end function test_real

  function test_real4(in_real) result(out_int)
    real(kind=4)  :: in_real
    integer       :: out_int
    out_int = 2
  end function test_real4

  function test_real8(in_real) result(out_int)
    real(kind=8)  :: in_real
    integer       :: out_int
    out_int = 3
  end function test_real8

end module m_test
