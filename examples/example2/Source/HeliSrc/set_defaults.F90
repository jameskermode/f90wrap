!>======================================================================
!!       This subroutine is used to set the default values of 
!!          various switches at the beginning of program      
!!======================================================================

      subroutine set_defaults(Solver)

      use defineAllProperties
      implicit none

!=======================================================================
!                         EXECUTABLE CODE
!=======================================================================

      type(SolverOptionsDef), intent(inout) :: Solver
      
!=======================================================================
!                    Solver option defaults
!=======================================================================

      Solver % AirframeVib                            = .true.
      Solver % FET_qddot                              = .false.
      Solver % FET_Response                           = .false.
      Solver % FusHarm                                = .false.
      Solver % STore_FET_ResponseJac                  = .false.
      
!=======================================================================
!                      End of operations
!=======================================================================

      return
      end subroutine set_defaults
