
module m_out_test
  implicit none
  private

  public :: out_scalar_int1,out_scalar_int2
  public :: out_scalar_int4,out_scalar_int8
  public :: out_scalar_real4,out_scalar_real8
  public :: out_array_int4,out_array_int8
  public :: out_array_real4,out_array_real8

contains

  function out_scalar_int1() result(output)
    integer(kind=1) :: output
    output=1
  end function

  function out_scalar_int2() result(output)
    integer(kind=2) :: output
    output=2
  end function

  function out_scalar_int4() result(output)
    integer(kind=4) :: output
    output=4
  end function

  function out_scalar_int8() result(output)
    integer(kind=8) :: output
    output=8
  end function

  function out_scalar_real4() result(output)
    real(kind=4) :: output
    output=4
  end function

  function out_scalar_real8() result(output)
    real(kind=8) :: output
    output=8
  end function

  function out_array_int4() result(output)
    integer(kind=4) :: output(1)
    output=4
  end function

  function out_array_int8() result(output)
    integer(kind=8) :: output(1)
    output=8
  end function

  function out_array_real4() result(output)
    real(kind=4) :: output(1)
    output=4
  end function

  function out_array_real8() result(output)
    real(kind=8) :: output(1)
    output=8
  end function

end module m_out_test


