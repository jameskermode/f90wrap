module KIMDispersion_Horton_module
    use KIMDispersionEquation_module
    implicit none

    type, extends(KIMDispersionEquation) :: KIMDispersion_Horton
        type(OptionsType) :: options
        type(SpeciesDataType), allocatable :: species(:)
        type(SpeciesDataType), allocatable :: spec_dat(:)
        type(GeneralDataType) :: general_dat
        type(EquilDataType) :: equil_dat
    contains
        procedure :: initialize => initialize_KIMDispersion_Horton
        procedure :: dispersion_equation => dispersion_equation_KIMDispersion_Horton
    end type KIMDispersion_Horton

contains

    subroutine initialize_KIMDispersion_Horton(this, options, species, spec_dat, general_dat, equil_dat)
        class(KIMDispersion_Horton), intent(inout) :: this
        type(OptionsType), intent(in) :: options
        type(SpeciesDataType), intent(in) :: species(:)
        type(SpeciesDataType), intent(in) :: spec_dat(:)
        type(GeneralDataType), intent(in) :: general_dat
        type(EquilDataType), intent(in) :: equil_dat

        this%options = options
        this%species = species
        this%spec_dat = spec_dat
        this%general_dat = general_dat
        this%equil_dat = equil_dat
    end subroutine initialize_KIMDispersion_Horton

    function dispersion_equation_KIMDispersion_Horton(this, kr, r_indx) result(dispersion_equation)
        class(KIMDispersion_Horton), intent(inout) :: this
        complex(8), intent(in) :: kr
        integer, intent(in) :: r_indx
        complex(8) :: dispersion_equation
        complex(8) :: nom, denom
        real(8) :: ky, om_prime
        integer :: spec

        this%general_dat%kperp = sqrt(this%general_dat%ks**2 + kr**2)
        ky = this%general_dat%ks(r_indx)
        om_prime = this%options%omega - this%general_dat%om_E(r_indx)

        dispersion_equation = cmplx(0.0, 0.0)
        nom = cmplx(0.0, 0.0)
        denom = cmplx(0.0, 0.0)

        do spec = 1, size(this%species)
            this%spec_dat(spec)%om_n = ky * sol / &
                (this%spec_dat(spec)%charge * this%equil_dat%B0(r_indx) &
                * this%spec_dat(spec)%n(r_indx)) * this%spec_dat(spec)%T(r_indx) &
                * ev * this%spec_dat(spec)%dndr(r_indx)

            this%spec_dat(spec)%om_T = ky * sol / &
                (this%spec_dat(spec)%charge * this%equil_dat%B0(r_indx)) &
                * this%spec_dat(spec)%dTdr(r_indx) * ev

            this%spec_dat(spec)%z = om_prime / &
                (sqrt(2.0) * this%general_dat%kp(r_indx) &
                * this%spec_dat(spec)%vT(r_indx))

            this%spec_dat(spec)%W = &
                (this%spec_dat(spec)%om_n - om_prime + &
                this%spec_dat(spec)%om_T * (this%spec_dat(spec)%z**2 + 0.5)) &
                * this%spec_dat(spec)%z / om_prime &
                * plasma_disp(this%spec_dat(spec)%z) &
                + this%spec_dat(spec)%om_T * this%spec_dat(spec)%z**2 / om_prime

            this%spec_dat(spec)%nom = &
                this%spec_dat(spec)%lambda_D(r_indx)**-2 * &
                (this%spec_dat(spec)%W - 1.0 - this%spec_dat(spec)%om_T &
                * this%spec_dat(spec)%z / om_prime &
                * plasma_disp(this%spec_dat(spec)%z))

            this%spec_dat(spec)%denom = &
                this%spec_dat(spec)%W * this%spec_dat(spec)%vT(r_indx)**2 / &
                (this%spec_dat(spec)%omega_c(r_indx)**2 &
                * this%spec_dat(spec)%lambda_D(r_indx)**2)

            nom = nom + this%spec_dat(spec)%nom
            denom = denom + this%spec_dat(spec)%denom
        end do

        nom = nom - this%general_dat%kp(r_indx)**2
        dispersion_equation = nom / (1.0 + denom) &
            - this%general_dat%kperp(r_indx)**2
    end function dispersion_equation_KIMDispersion_Horton

end module KIMDispersion_Horton_module
