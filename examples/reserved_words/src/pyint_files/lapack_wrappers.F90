!======================================================================
!These subroutines provide interfaces to the netlib canned procedures
!that perform standard linear-algebra operations on matrices 
!======================================================================

! 1 - matrix inversion                                      DGECO/DGEDI
! 2 - simultaneous linear equation solver                   DGESV

      subroutine INVERt(A,LDA,N,RCOND,DETERM)

      implicit none

!======================================================================
! Inputs
!======================================================================

      integer        , intent(in)   :: N, LDA

!======================================================================
! Input/Output
!======================================================================
      
      real , intent(inout):: A(LDA, N)

!======================================================================
! Outputs
!======================================================================
      
      real , intent(inout):: rcond, determ
      
!======================================================================
! Local variables
!======================================================================

      INTEGER                       :: JOB, IPVT(N)
      real              :: DET(2)
      real              :: WORK2(N) ,WORK(N)     

!======================================================================
! Begin executable code
!======================================================================
      
!set job to 11 to get both determinant and inverse
      JOB=11      

!double precision version
!#ifdef usedouble
      write(*,*) 'calling inverse'
      CALL sGECO(A,LDA,N,IPVT,RCOND,WORK2)
      CALL sGEDI(A,LDA,N,IPVT,DET,WORK,JOB)

      DETERM=DET(1) * 10.0**DET(2)
      
      return
      end subroutine
      
! !======================================================================
! ! Simultaneous linear equation solver
! !======================================================================
      
!       subroutine DGESV2(N, A, LDA, B, X)
!       implicit none

! !======================================================================
! ! Inputs
! !======================================================================
      
!       integer        , intent(in) :: N,LDA
!       double precision , intent(in) :: A(LDA,N),B(N)

! !======================================================================
! ! Outputs
! !======================================================================

!       double precision , intent(out):: X(N)      

! !======================================================================
! ! Local variables
! !======================================================================

!       integer                    :: ii,jj,IPIV(N),info
!       double precision             :: ANET(N,N),BNET(N,1)

! !     COPY THE RIGHT HAND SIDE TO THE NEW STORAGE AREA

!       DO II = 1, N
!          BNET( II ,1 ) = B( II )
!          DO JJ = 1, N
!             ANET( II, JJ ) = A ( II, JJ )
!          ENDDO
!       ENDDO

! !     CALCULATE THE SOLUTION OF THE LINEAR SYSTEM
! !#ifdef usedouble
!       CALL DGESV( N, 1, ANET, N, IPIV, BNET, N, INFO )
! !#else
! !      call sgesv(N,1,ANET,N,IPIV,BNET,N,INFO)
! !#endif

! !     CHECK THE STATUS OF THE NETLIB SUBROUTINE
!       IF(INFO.NE.0) THEN
!          WRITE(6,*) ' WARNING: PROBLEM WITH DGESV OUTPUT IN netkey1'
         
!       ENDIF

! !     COPY THE SOLUTION TO THE OUTPUT VECTOR

!       DO II = 1, N
!          X ( II ) = BNET ( II ,1 )
!       ENDDO
!       return
!       end subroutine dgesv2