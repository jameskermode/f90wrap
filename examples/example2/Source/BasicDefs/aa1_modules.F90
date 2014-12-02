!=======================================================================
!Author  : Ananth Sridharan
!=======================================================================

!=======================================================================
!purpose:store some memory bounds in an easily accessible location. 
!        This is to provide some array bounds during compilation, with
!        the understanding that the integers here will be changed 
!        AT MOST ~ 1,2 times a year with every repository update
!=======================================================================

	module constant_parameters

         use precision

         parameter (MAXTEL  =  12)                    ! max # time elements
         parameter (maxnst  = 200)                    ! max # states
         parameter (MXCV    =   5)                    ! max # controls 
         parameter (MXRt    =   3)                    ! max # rotors / aircraft
         parameter (MXBD    =   4)                    ! max # blades / rotor
         parameter (mxmodes =  15)                    ! max # modes  / blade
         parameter (MXFEM   =  15)                    ! max # elmnts / blade
         parameter (mxndf   =8*mxfem+6)               ! max # dof    / blade
         parameter (MXNG    =   8)                    ! max # quadrature pts
         parameter (MXNGPSI =   4)                    ! max # azimuthal quadrature pts
         parameter (LAZIM   =  12*MAXTEL+1)           ! max # azim. points 
         parameter (MXL     =   6)                    ! max # dyn. inflow states
         parameter (mxtv    = 500)                    ! max # trim variables
         parameter (mxafoil =   6)                    ! max # airfoils
         parameter(MXNS     = mxng*mxfem)             ! max # quadrature points

 	end module

!=======================================================================
!Replaces common blocks GSQUAD, GAUSS and AZIM
!=======================================================================

      module gaussian

         use precision
         use constant_parameters

!=======================================================================
!The following are integration weights and locations for the (0,1) domain
!=======================================================================

         integer        :: ng, ngpsi

         real(kind=rdp) :: ECinv(mxng,mxng), xg(mxng),                  &
                           FCinv(mxng,mxng), wg(mxng)

!for azimuth
         real(kind=rdp) :: xgpsi(mxngpsi), wgpsi(mxngpsi)

      end module
