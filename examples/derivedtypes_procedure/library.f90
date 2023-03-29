module test
  implicit none

  private
  public :: atype, btype
  public :: create, asum
  public :: asum_class

  type :: atype
     integer, allocatable        :: array(:)
  contains
     procedure                   :: p_create => create_class
     procedure                   :: p_asum => asum_class
     procedure                   :: p_asum_2 => asum_class
     procedure                   :: asum_class
     procedure                   :: p_reset => assignment_value
     generic                     :: assignment(=) => p_reset
  end type atype

  type :: btype
     integer                     :: array(3)
  contains
     procedure                   :: p_asum => bsum_class
  end type btype

contains

   subroutine create(self, n)
      implicit none
      type(atype),intent(inout)  :: self
      integer                    :: n
      !
      if (allocated(self%array)) deallocate(self%array)
      allocate(self%array(n))
      !
   end subroutine

   subroutine create_class(self, n)
      implicit none
      class(atype),intent(inout) :: self
      integer                    :: n
      !
      call create(self, n)
      !
   end subroutine

   function asum(self)
      implicit none
      type(atype),intent(in)     :: self
      real                       :: asum
      !
      asum = sum(self%array)
      !
   end function

   function asum_class(self)
      implicit none
      class(atype),intent(in)    :: self
      real                       :: asum_class
      !
      asum_class = asum(self)
      !
   end function

   subroutine assignment_value(self, value)
      implicit none
      class(atype),intent(inout) :: self
      integer,intent(in)         :: value
      !
      self%array(:) = value
      !
   end subroutine

   function bsum_class(self)
      implicit none
      class(btype),intent(in)    :: self
      real                       :: bsum_class
      !
      bsum_class = sum(self%array)
      !
   end function

end module
