!======================================================================
!                 Original author Shreyas Ananthan
!                 Modified by     Ananth Sridharan
!======================================================================
      module precision
   !Kind parameters for real variables
!      integer, parameter      :: float =kind(1.e0) ! single precision
!      integer, parameter      :: double=kind(1.D0) ! double precision

!Uncomment lines below to enable switching.. right now, using rdp=double
!      #ifdef USE_SINGLE_PRECISION
!         integer, parameter  :: rdp=4
!          zero = 0.0 
!      #else

!must specify # bytes integer explicitly here for f2c and f2py
          integer, parameter  :: rdp=8
!      #endif
      real(kind=rdp)          ::  ZERO,ONE,HALF,TWO,THREE,FOUR,SIX,     &
                                  EIGHT,PI,TWOPI, D2R, R2D, XK2FPS,     &
                                  LB2N, FTLB2NM, ONE80,FT2M, GSI, GFPS, &
                                  Three60, IN2FT, FIVE

      end module precision
