module parameters

implicit none
private
!public :: idp, isp, do_array_stuff
public :: idp, isp

INTEGER, PARAMETER :: idp=kind(1d0)
INTEGER, PARAMETER :: isp=kind(1e0)

contains
    subroutine do_array_stuff()
    end subroutine do_array_stuff

end module parameters

