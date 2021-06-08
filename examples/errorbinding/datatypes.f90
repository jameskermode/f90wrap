! ==============================================================================
module datatypes
use parameters, only: idp, isp
implicit none
private 
public :: typewithprocedure, constructor_typewithprocedure, info_typewithprocedure
type :: typewithprocedure
   REAL(idp) :: a
   INTEGER(4) :: n
 contains
   procedure, public :: init => init_procedure
   procedure, public :: info => info_procedure
end type typewithprocedure
contains

  subroutine init_procedure(this, a, n)
    class(typewithprocedure), intent(inout) :: this
    REAL(idp),   intent(in   ) :: a
    INTEGER(4),  INTENT(in) :: n

    this%a = a
    this%n = n
  end subroutine init_procedure

  subroutine info_procedure(this, lun)
    class(typewithprocedure), intent(inout) :: this
    INTEGER(4),  INTENT(in) :: lun

    write(lun,*) ' a = ', this%a, 'n = ', this%n
  end subroutine info_procedure

  subroutine constructor_typewithprocedure(this, a, n)
    type(typewithprocedure), intent(inout) :: this
    REAL(idp),   intent(in   ) :: a
    INTEGER(4),  INTENT(in) :: n

    call this%init(a,n)
  end subroutine constructor_typewithprocedure


  subroutine info_typewithprocedure(this, lun)
    type(typewithprocedure), intent(inout) :: this
    INTEGER(4),  INTENT(in) :: lun

    call this%info(lun)
  end subroutine info_typewithprocedure

end module datatypes
