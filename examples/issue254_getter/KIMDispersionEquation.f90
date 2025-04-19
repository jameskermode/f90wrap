module KIMDispersionEquation_module
    implicit none

    real(8), parameter :: pi = 3.1415
    real(8), parameter :: e_mass = 9.1094e-28
    real(8), parameter :: p_mass = 1.6726e-24
    real(8), parameter :: e_charge = 4.8032e-10
    real(8), parameter :: ev = 1.6022e-12
    real(8), parameter :: sol = 29979245800.0
    real(8), parameter :: kB = 1.380649e-16

    ! Type for options
    type :: OptionsType
        real(8) :: omega
        ! Add other fields as needed
    end type OptionsType

    ! Type for species-specific data
    type :: SpeciesDataType
        real(8) :: charge
        real(8), allocatable :: n(:)
        real(8), allocatable :: T(:)
        real(8), allocatable :: dndr(:)
        real(8), allocatable :: dTdr(:)
        real(8), allocatable :: vT(:)
        real(8), allocatable :: lambda_D(:)
        real(8), allocatable :: omega_c(:)
        real(8) :: om_n
        real(8) :: om_T
        complex(8) :: z
        complex(8) :: W
        complex(8) :: nom
        complex(8) :: denom
    end type SpeciesDataType

    ! Type for general data
    type :: GeneralDataType
        real(8), allocatable :: ks(:)
        real(8), allocatable :: om_E(:)
        real(8), allocatable :: kp(:)
        real(8), allocatable :: kperp(:)
    end type GeneralDataType

    ! Type for equilibrium data
    type :: EquilDataType
        real(8), allocatable :: B0(:)
    end type EquilDataType

    ! Abstract base type
    type, abstract :: KIMDispersionEquation
    contains
        procedure(initialize_interface), deferred :: initialize
        procedure(dispersion_equation_interface), deferred :: dispersion_equation
    end type KIMDispersionEquation

    ! Abstract interface for the initialize method
    abstract interface
        subroutine initialize_interface(this, options, species, spec_dat, general_dat, equil_dat)
            import :: KIMDispersionEquation, OptionsType, SpeciesDataType, GeneralDataType, EquilDataType
            class(KIMDispersionEquation), intent(inout) :: this
            type(OptionsType), intent(in) :: options
            type(SpeciesDataType), intent(in) :: species(:)
            type(SpeciesDataType), intent(in) :: spec_dat(:)
            type(GeneralDataType), intent(in) :: general_dat
            type(EquilDataType), intent(in) :: equil_dat
        end subroutine initialize_interface
    end interface

    ! Abstract interface for the dispersion_equation method
    abstract interface
        function dispersion_equation_interface(this, kr, r_indx) result(dispersion_equation)
            import :: KIMDispersionEquation
            class(KIMDispersionEquation), intent(inout) :: this
            complex(8), intent(in) :: kr
            integer, intent(in) :: r_indx
            complex(8) :: dispersion_equation
        end function dispersion_equation_interface
    end interface

contains

    function plasma_disp(x) result(y)
        complex(8), intent(in) :: x
        complex(8) :: y

        y = 1.0d0
    end function

end module KIMDispersionEquation_module
