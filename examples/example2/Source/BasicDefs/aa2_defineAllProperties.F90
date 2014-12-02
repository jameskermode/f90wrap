!=======================================================================
!                 PREFACE : DEFINEALLPROPERTIES
!=======================================================================
!        Some derived types contain other derived types, which must be 
!     defined before they can be used.  Therefore, the order in which 
!     the files are included is important.  See comments to the code for 
!     details on the required order.
!=======================================================================

#define type(x) TYPE(x), target

      module defineAllProperties 

         use constant_parameters
         
!=======================================================================
!     This derived type remembers user input options for trim,
!                 linearization and time marching
!=======================================================================

         TYPE :: SolverOptionsDef
            
!=======================================================================
!                          Read trim options
!=======================================================================

            logical           ::       TrimSwitch, UpdateGuess,         &
                                    DeltaAirloads
            logical           ::      LinrzSwitch
            logical           ::  TimeMarchSwitch
            logical           ::   FreeWakeSwitch
            logical           :: WindTunnelSwitch
            logical           :: RigidBladeSwitch
            logical           :: FET_qddot, FET_Response
            logical           :: Store_FET_ResponseJac, FET_ResponseJacAvail
            logical           :: AirframeVib, FusHarm
            logical           :: AxialDOF, composite_coupling

            integer           :: TrimTechnique
            integer           :: TrimSweepOption, NTimeElements
            integer           :: NBladeHarm, NbladeModes
            integer           :: ModeOrder(mxmodes)
            integer           :: NCosInflowHarm, NmaxInflowPoly, linflm
            integer           :: LinrzPts
            integer           :: ControlHistOption, NRevolutions, NAzim
            integer           :: NTimeSteps
            integer           :: Nred, Nred2
      
            real(kind=rdp)    :: TrimConvergence, IntegError
            real(kind=rdp)    :: LinrzPert
            real(kind=rdp)    :: ControlAmplitude, ControlFrequency
            real(kind=rdp)    :: Jac(15,15), jac2(15,15)

         END TYPE SolverOptionsDef

      end module defineAllProperties

!=======================================================================
