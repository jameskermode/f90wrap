MODULE mcyldnad
CONTAINS
!>    Computes cylinder volume from 2 input variables (radius, height)
SUBROUTINE cyldnad(vol, radius, height)
        
  USE Dual_Num_Auto_Diff
  TYPE (DUAL_NUM), PARAMETER:: PI=DUAL_NUM(3.141592653589793D0,0.D0)
  TYPE (DUAL_NUM), INTENT(out):: vol
  TYPE (DUAL_NUM), INTENT(in) :: radius, height
        
  vol = PI * radius**2 * height

END SUBROUTINE cyldnad
END MODULE mcyldnad

