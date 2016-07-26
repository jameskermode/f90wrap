
module io
    
    implicit none

    private

    type, public :: keyword
       character(len=10) :: key
       character(len=3)  :: typ
       character(len=10) :: description
    end type keyword

    public :: io_freeform_open

contains

    subroutine io_freeform_open(filename,keywords)
    
        implicit none
    
        character(len=*),            intent(in) :: filename
        !type(keyword), intent(in) :: keywords
        type(keyword), optional ,dimension(:), intent(in) :: keywords
    
        
        return
    
    end subroutine io_freeform_open

end module
