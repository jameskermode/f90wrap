
module testextends_mod


PUBLIC

	! -----------------------------------------------
	type Superclass
		! IN: Ask subroutine to stop in the middle.
		integer :: stop_at = -1		! -1 --> don't stop
	end type Superclass

	type, extends(Superclass) :: Subclass1
		integer :: nl
	end type

	type, extends(Superclass) :: Subclass2
		integer :: nl
	end type

end module
