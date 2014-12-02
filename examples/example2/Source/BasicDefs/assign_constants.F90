!=======================================================================
!     assign universal constants and quadrature weights
!=======================================================================

      subroutine assign_constants()
      use precision
      use gaussian
      implicit none

!=======================================================================
!local constants (one-time use)
!=======================================================================

      real(kind=rdp) :: KT2KMPH, TWOPT2, KMPH2MPS

      real(kind=rdp), allocatable :: Etemp(:,:), Ftemp(:,:) 

!=======================================================================
!#ifdef usedouble
!=======================================================================

      ZERO     = 0.d0  ;  ONE = 1.d0;    HALF = 0.5d0;   TWO = 2.d0
      ONE80    = 180.d0; FT2M = 0.3048d0; GSI = 9.8d0; THREE = 3.d0
      TWOPT2   = 2.2d0 ; FIVE = 5.0d0
      Three60  = 360.0d0  

      KMPH2MPS = 5.d0/18.d0;
      KT2KMPH  = 1.852d0;

!=======================================================================
!#else
!=======================================================================

!      ZERO = 0.e0; ONE = 1.e0; HALF = 0.5e0; TWO = 2.e0; THREE = 3.e0
!      ONE80= 180.e0; FT2M = 0.3048d0; GSI = 9.8e0; KT2KMPH = 1.852e0
!      KMPH2MPS = 5.e0/18.e0; TWOPT2 = 2.2e0

!=======================================================================
!#endif
!=======================================================================

      GFPS        = GSI/FT2M

!=======================================================================
!let's go back to primary school now
!=======================================================================

      FOUR        = TWO*TWO
      SIX         = TWO*THREE
      EIGHT       = TWO*FOUR
      
!=======================================================================
!and now, to high school 
!=======================================================================

      PI          = FOUR*ATAN(ONE)         !   pi
      TWOPI       = TWO*PI                 ! 2 pi
      D2R         = PI/ONE80               ! deg to radians conversion factor
      R2D         = ONE80/PI               ! radians to degrees
      XK2FPS      = KT2KMPH*KMPH2MPS/FT2M  ! knots to feet per second
      LB2N        = GSI/TWOPT2             ! pounds of force to Newtons
      FTLB2NM     = FT2M*LB2N              ! ft-lb of moment to Newton-meters
      IN2FT       = one/(three*four)       ! inches to feet of length

!=======================================================================
!            Generate Gauss-Quadrature Coefficients
!                "NG" - ORDER OF GAUSS QUAD 
!=======================================================================

      NG          = MXNG                   ! # spatial quadrature points
      ngpsi       = mxngpsi                ! # azimuthal quadrature points

!=======================================================================
!           Generate azimuthal quadrature points
!=======================================================================

!deallocate work arrays (not used)
      if(allocated(Etemp)) deallocate(Etemp)
      if(allocated(Ftemp)) deallocate(Ftemp)

!allocate work arrays 
      allocate(Etemp(ngpsi,ngpsi), Ftemp(ngpsi,ngpsi))

!generate quadrature points
!      call gqgen2( ngpsi, Etemp, Ftemp, xgpsi, wgpsi)

!deallocate work arrays
      if(allocated(Etemp)) deallocate(Etemp)
      if(allocated(Ftemp)) deallocate(Ftemp)
      
!=======================================================================
!                       End of operations
!=======================================================================

      return
      end subroutine assign_constants
