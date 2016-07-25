
module cell
    
    implicit none

    private

    type, public::unit_cell
        integer         ::num_species
        character(len=8)::species_symbol
    end type unit_cell

    public::cell_dosomething

contains

    subroutine cell_dosomething(cell, num_species, species_symbol)
    
        implicit none
    
        type(unit_cell), intent(inout)  ::cell
        integer, intent(in)             ::num_species
        character(len=8), intent(in)    ::species_symbol
    
        cell%num_species=num_species
        cell%species_symbol=species_symbol
        
        return
    
    end subroutine cell_dosomething

end module
