!*************************************************************************
!* Dual Number Automatic Differentiation (DNAD) of Fortran Codes
!*----------------------------------------------------------------
!* COPYRIGHT (c) Wenbin Yu, All rights reserved, you are free to copy,
!* modify or translate this code to other languages such as c/c++. If
!* you find a bug please let me know through wenbinyu.heaven@gmail.com. If
!* you added new functions and want to share with others, please let me know too.
!* You are welcome to share your successful stories with us through
!* http://groups.google.com/group/hifi-comp.
!******************************************************************************
!* Simple Instruction:
!*---------------------
!* This is the procedure file: define the procedures (functions/subroutines)
!* needed for overloading intrinsic functions and operators. If a function
!* or operation is not defined (unlikely), you need to create a corresponding
!* interface in header file (DNADHeaders.f90) and a corresponding
!* function/subroutine in DNAD.f90.
!* If you are uncertain whether I did things in the correct way or not, you can
!* create a new module with the same definition of the type DUAL_NUM, but delete
!* all the interfaces and procedures first. Then compile it with your analysis
!* codes after you changed all the definitions of real numbers to be dual numbers.
!* The compiler will complain about some operations and functions are not defined.
!* Then you do it one by one by looking at my original module, copying the needed
!* interfaces/procedures into your module. This way, you can also get a leaner
!* model and use those only needed for your analysis code.
!*********************************************************************************
!* Acknowledgements
!*-------------------
!* The development of DNAD is supported, in part, by the Chief Scientist
!* Innovative Research Fund at AFRL/RB WPAFB, and by Department of Army
!* SBIR (Topic A08-022) through Advanced Dynamics Inc. The views and
!* conclusions contained herein are those of the authors and should not be
!* interpreted as necessarily representing the official policies or endorsement,
!* either expressed or implied, of the funding agency.
!**********************************************************************************
!* Basic idea
!*-------------------------------------
!* Carry out Automatic Differentiation using the arithmetic of dual numbers.
!* It is supposed to be more efficient and more accurate than complex-step
!* approximation. What one needs to do is to create a new data type according
!* to the rules of dual numbers, which is equivalent to create a quasi-complex
!* numbers. And overload all the intrinsic operators and functions for
!* such a type.
!* For simplicity and memoric convenience, we call a dual number as a dual (D),
!* a double precision real number as a real (R) and a single precision real
!* number as a single (S), and an integer number as an integer (I). We also
!* use M to denote a 2D array, V to denote a vector, M3 to denote a 3D arry.
!* By defining functions/subroutines to the ELEMENTAL, the inputs/outputs
!* can be automatically overloaded for vectors and matrices of the same shape
!**********************************************************************************
!* To AD a Fortran code use DNAD
!* Step 0: compile DNAD.f90 to be DNAD.o
!* Step 1: Replace all the definitions of real numbers with dual numbers
!*		For example
!*  			replace REAL(8) :: x
!*   			with  TYPE(DUAL_NUM) :: x
!*  			replace REAL(8), PARAMETER:: ONE=1.0D0
!*   			with TYPE(DUAL_NUM),PARAMETER::ONE=DUAL_NUM(1.0D0,0.D0)
!* Step 2: Insert USE Dual_Num_Auto_Diff right after Module/Function/Subroutine/Program
!*         statements used TYPE(DUAL_NUM)
!* Step 3: Change IO commands correspondingly if the code does not use free formatting
!*         read and write (can be automated by written some general-purpose utility subroutines)
!* Step 4: Recompile the source along with DNAD.o
!* The whole process can be automated, and even manually it only takes just a few minutes
!* for most real analysis codes, although step 3 is code dependent.
!*****************************************************************************************
!* Example
!* The analysis version of the code computing the area of a circle with input radius
!* PROGRAM CircleArea
!*	REAL(8),PARAMETER:: PI=3.141592653589793D0
!*	REAL(8):: radius, area
!*	READ(*,*) radius
!*	Area=PI*radius**2
!*	WRITE(*,*) "AREA=", Area
!* END PROGRAM CircleArea
!*Input: 5
!*Output: AREA=78.5398163397448
!*---------------------------------------------------------------------------------------
!* The AD version of the code computing the area of a circle and sensitivity with input radius
!*PROGRAM CircleArea
!*	USE DNAD
!*	TYPE (DUAL_NUM),PARAMETER:: PI=DUAL_NUM(3.141592653589793D0,0.D0)
!*	TYPE (DUAL_NUM):: radius,area
!*	READ(*,*) radius
!*	Area=PI*radius**2
!*	WRITE(*,*) "AREA=",Area
!*END PROGRAM CircleArea
!* Input 5, 1
!* Output: AREA=78.5398163397448, 31.4159265358979
!*--------------------------------------------------------
!*****************************************************************************************
!* Possible Mistakes to Avoid
!* 11/23/2011: DNAD should always compute the same functional value. However, in very rare situations,
!* sensitivities are not computed as expected. One such case has been identified with
!* dgemv in the blas package. The main reason is that dgemv skip some calculations due to
!* the fact of multiplying zero. However, sometimes, the functional value is zero, but
!* the sensitivity is not zero. In DNAD the comparision is done only between functional values
!* hence is sensitivity calculation is also avoid altogether. For example, the following piece of code
!*                IF( X( JX ).NE.ZERO )THEN
!*                  TEMP = ALPHA*X( JX )
!*                  DO 50, I = 1, M
!*                     Y( I ) = Y( I ) + TEMP*A( I, J )
!*   50             CONTINUE
!*              END IF
!* the solution is to either to rewrite dgemv or assign epsilon(0.0d0) to x(jx) if the functional value
!* while the corresponding derivatives are not zero.
!************************************************************************************************************

MODULE Dual_Num_Auto_Diff

IMPLICIT NONE

INTEGER(2), PUBLIC,PARAMETER:: NDV_AD=2 ! number of design variables

PRIVATE


INTEGER, PARAMETER:: SNG_AD=SELECTED_REAL_KIND(6)  ! single precision
INTEGER, PARAMETER:: DBL_AD=SELECTED_REAL_KIND(15) ! double precision
REAL(DBL_AD)      ::negative_one=-1.0d0

TYPE, PUBLIC:: DUAL_NUM
 	                              ! make this private will create difficulty to use the original write/read commands,
                                  ! hence x_ad_ and xp_ad_ are variables which can be accessed using D%x_ad_ and
    					          ! D%xp_ad_ in other units using this module in which D is defined as TYPE DUAL_NUM.
	REAL(DBL_AD)::x_ad_           ! functional value
	REAL(DBL_AD)::xp_ad_(NDV_AD)  ! derivative

END TYPE DUAL_NUM

#include "DNADHeaders.h"


CONTAINS

!*********Begin: functions/subroutines for overloading operators

!******* Begin: (=)
!---------------------
	!** dual=integer: 	! <u,up>=<a,0.D0>
	ELEMENTAL SUBROUTINE ASSIGN_DI(u,n)
         TYPE (DUAL_NUM), INTENT(OUT)::u
         INTEGER, INTENT(IN)::n

 !         u=DUAL_NUM(n,0.d0)    ! It is shown that this approach is much slower than the new approach
         u%x_ad_= REAL(n,DBL_AD) ! REAL(n,DBL_AD) is faster than let the code do the conversion.
         u%xp_ad_=0.0D0

	END SUBROUTINE ASSIGN_DI

	!** dual=real: <u,up>=<a,0.D0>
	ELEMENTAL SUBROUTINE ASSIGN_DR(u,n)
         TYPE (DUAL_NUM), INTENT(OUT)::u
         REAL(DBL_AD), INTENT(IN)::n

          u%x_ad_=n
          u%xp_ad_=0.0D0


	END SUBROUTINE ASSIGN_DR

	!** dual=single
	! <u,up>=<n,0.D0>
	ELEMENTAL SUBROUTINE ASSIGN_DS(u,n)
         TYPE (DUAL_NUM), INTENT(OUT)::u
         REAL(SNG_AD), INTENT(IN)::n

	     u%x_ad_=REAL(n,DBL_AD)
         u%xp_ad_=0.0D0

	END SUBROUTINE ASSIGN_DS

	!** integer=dual
	! n=u%x_ad_
	ELEMENTAL SUBROUTINE ASSIGN_ID(n,u)
         TYPE (DUAL_NUM), INTENT(IN)::u
         INTEGER, INTENT(OUT)::n

         n=u%x_ad_

	END SUBROUTINE ASSIGN_ID

!******* End: (=)
!---------------------


!******* Begin: (+)
!---------------------


	!dual=+<v,vp>
	!-----------------------------------------
    ELEMENTAL FUNCTION ADD_D(u) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
		 TYPE (DUAL_NUM)::res

         res=u ! It is faster than assigning component wise

	END FUNCTION ADD_D


	! <u,up>+<v,vp>=<u+v,up+vp>
	!-----------------------------------------
    ELEMENTAL FUNCTION ADD_DD(u,v) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u,v
		 TYPE (DUAL_NUM)::res

         res%x_ad_  = u%x_ad_+v%x_ad_
         res%xp_ad_ = u%xp_ad_+v%xp_ad_

	END FUNCTION ADD_DD

    ! dual+integer
	! <v,vp>+<n,0>=<n+v,vp>
	!-----------------------------------------
	ELEMENTAL FUNCTION ADD_DI(v,n) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::v
		 INTEGER,INTENT(IN)::n
		 TYPE (DUAL_NUM)::res

         res%x_ad_  = REAL(n,DBL_AD)+v%x_ad_
         res%xp_ad_ = v%xp_ad_

	END FUNCTION ADD_DI

	! dual +real
	! <v,vp>+<n,0>=<n+v,vp>
	!-----------------------------------------
	ELEMENTAL FUNCTION ADD_DR(v,n) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::v
		 REAL(DBL_AD),INTENT(IN)::n
		 TYPE (DUAL_NUM)::res

         res%x_ad_  = n+v%x_ad_
         res%xp_ad_ = v%xp_ad_

	END FUNCTION ADD_DR

	!-----------------------------------------
    ! dual+single
	! <v,vp>+<n,0>=<n+v,vp>
	!-----------------------------------------
	ELEMENTAL FUNCTION ADD_DS(v,n) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::v
		 REAL(SNG_AD),INTENT(IN)::n
		 TYPE (DUAL_NUM)::res

         res%x_ad_  = REAL(n,DBL_AD)+v%x_ad_
         res%xp_ad_ = v%xp_ad_

	END FUNCTION ADD_DS
	!-----------------------------------------


    !-----------------------------------------
	!an integer+ dual number
    ! <n,0>+<v,vp>=<n+v,vp>
	!-----------------------------------------
    ELEMENTAL FUNCTION ADD_ID(n,v) RESULT(res)
         INTEGER,INTENT(IN)::n
		 TYPE (DUAL_NUM), INTENT(IN)::v
		 TYPE (DUAL_NUM)::res

         res%x_ad_  = REAL(n,DBL_AD)+v%x_ad_
         res%xp_ad_ = v%xp_ad_

	END FUNCTION ADD_ID


    !-----------------------------------------
	! real + dual
	! <n,0>+<v,vp>=<n+v,vp>
	!-----------------------------------------
	ELEMENTAL FUNCTION ADD_RD(n,v) RESULT(res)
         REAL(DBL_AD),INTENT(IN)::n
		 TYPE (DUAL_NUM), INTENT(IN)::v
		 TYPE (DUAL_NUM)::res

         res%x_ad_  = n+v%x_ad_
         res%xp_ad_ = v%xp_ad_

	END FUNCTION ADD_RD

 	! single + dual
	! <n,0>+<v,vp>=<n+v,vp>
	!-----------------------------------------
	ELEMENTAL FUNCTION ADD_SD(n,v) RESULT(res)
         REAL(SNG_AD),INTENT(IN)::n
		 TYPE (DUAL_NUM), INTENT(IN)::v
		 TYPE (DUAL_NUM)::res

         res%x_ad_  = REAL(n,DBL_AD)+v%x_ad_
         res%xp_ad_ = v%xp_ad_

	END FUNCTION ADD_SD

!******* End: (+)
!---------------------


!******* Begin: (-)
!---------------------


	!-----------------------------------------
	! negate a dual number
	!-------------------------------------------------
    ELEMENTAL FUNCTION MINUS_D(u) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
		 TYPE (DUAL_NUM)::res

	     res%x_ad_ = -u%x_ad_
         res%xp_ad_= -u%xp_ad_

	END FUNCTION MINUS_D

	!-----------------------------------------
	! <u,up>-<v,vp>=<u-v,up-vp>
	!-------------------------------------------------
    ELEMENTAL FUNCTION MINUS_DD(u,v) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u,v
         TYPE (DUAL_NUM)::res

	     res%x_ad_ = u%x_ad_-v%x_ad_
         res%xp_ad_= u%xp_ad_-v%xp_ad_

	END FUNCTION MINUS_DD

	!-----------------------------------------
	! dual number - integer
	! <u,up>-<n,0>=<u-n,up>
	!-------------------------------------------------
	ELEMENTAL FUNCTION MINUS_DI(u,n) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
		 INTEGER,INTENT(IN)::n
		 TYPE (DUAL_NUM)::res

	     res%x_ad_ = u%x_ad_-REAL(n,DBL_AD)
         res%xp_ad_= u%xp_ad_

	END FUNCTION MINUS_DI

	!-----------------------------------------
	! dual number - real
	! <u,up>-<n,0>=<u-n,up>
	!-------------------------------------------------
	ELEMENTAL FUNCTION MINUS_DR(u,n) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
		 REAL(DBL_AD),INTENT(IN)::n
		 TYPE (DUAL_NUM)::res

	     res%x_ad_ = u%x_ad_-n
         res%xp_ad_= u%xp_ad_

	END FUNCTION MINUS_DR

    !-----------------------------------------
	! dual number - single
	! <u,up>-<n,0>=<u-n,up>
	!-------------------------------------------------
	ELEMENTAL FUNCTION MINUS_DS(u,n) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
		 REAL(SNG_AD),INTENT(IN)::n
		 TYPE (DUAL_NUM)::res

	     res%x_ad_ = u%x_ad_-REAL(n,DBL_AD)
         res%xp_ad_= u%xp_ad_

	END FUNCTION MINUS_DS


	!-----------------------------------------
	! integer-dual number
	!-------------------------------------------------
	ELEMENTAL FUNCTION MINUS_ID(n,u) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
		 INTEGER,INTENT(IN)::n
		 TYPE (DUAL_NUM)::res

	     res%x_ad_ = REAL(n,DBL_AD)-u%x_ad_
         res%xp_ad_= -u%xp_ad_

	END FUNCTION MINUS_ID

	!-----------------------------------------
	! real-dual number
	!-------------------------------------------------
	ELEMENTAL FUNCTION MINUS_RD(n,u) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
		 REAL(DBL_AD),INTENT(IN)::n
		 TYPE (DUAL_NUM)::res

	     res%x_ad_ = n-u%x_ad_
         res%xp_ad_= -u%xp_ad_

	END FUNCTION MINUS_RD

    !-----------------------------------------
	! single-dual number
	! <n,0>-<u,up>=<n-u,-up>
	!-------------------------------------------------
	ELEMENTAL FUNCTION MINUS_SD(n,u) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
		 REAL(SNG_AD),INTENT(IN)::n
		 TYPE (DUAL_NUM)::res

	     res%x_ad_ =REAL(n,DBL_AD) - u%x_ad_
         res%xp_ad_=- u%xp_ad_

	END FUNCTION MINUS_SD

!******* END: (-)
!---------------------


!******* BEGIN: (*)
!---------------------

	!-----------------------------------------
	! <u,up>*<v,vp>=<u*v,up*v+u*vp>
	!----------------------------------------
	ELEMENTAL FUNCTION MULT_DD(u,v) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u,v
		 TYPE (DUAL_NUM)::res

		 res%x_ad_ = u%x_ad_*v%x_ad_
         res%xp_ad_= u%xp_ad_*v%x_ad_ + u%x_ad_*v%xp_ad_

	END FUNCTION MULT_DD


	!-----------------------------------------
	!  dual*integer
	! <u,up>*<n,0>=<u*n,up*n>
	!----------------------------------------
	ELEMENTAL FUNCTION MULT_DI(u,n) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u
         INTEGER,INTENT(IN)::n
		 TYPE (DUAL_NUM)::res

         res%x_ad_ = REAL(n,DBL_AD)*u%x_ad_
         res%xp_ad_= REAL(n,DBL_AD)*u%xp_ad_

	END FUNCTION MULT_DI

    !-----------------------------------------
	!  dual*real
	! <u,up>*<n,0>=<u*n,up*n>
	!----------------------------------------
	ELEMENTAL FUNCTION MULT_DR(u,n) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u
         REAL(DBL_AD),INTENT(IN)::n
		 TYPE (DUAL_NUM)::res

         res%x_ad_ = n*u%x_ad_
         res%xp_ad_= n*u%xp_ad_

	END FUNCTION MULT_DR


	!-----------------------------------------
	!  dual*single
	! <u,up>*<n,0>=<u*n,up*n>
	!----------------------------------------
	ELEMENTAL FUNCTION MULT_DS(u,n) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u
         REAL(SNG_AD),INTENT(IN)::n
		 TYPE (DUAL_NUM)::res

         res%x_ad_ = REAL(n,DBL_AD)*u%x_ad_
         res%xp_ad_= REAL(n,DBL_AD)*u%xp_ad_

	END FUNCTION MULT_DS


	!-----------------------------------------
	! integer*dual
	! <n,0>*<v,vp>=<n*v,n*vp>
	!----------------------------------------
	ELEMENTAL FUNCTION MULT_ID(n,v) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::v
         INTEGER,INTENT(IN)::n
		  TYPE (DUAL_NUM)::res


         res%x_ad_ = REAL(n,DBL_AD)*v%x_ad_
         res%xp_ad_= REAL(n,DBL_AD)*v%xp_ad_

	END FUNCTION MULT_ID

	!-----------------------------------------
	! real* dual
	! <n,0>*<u,up>=<u*n,up*n>
	!----------------------------------------
    ELEMENTAL FUNCTION MULT_RD(n,u) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u
		 TYPE (DUAL_NUM)::res
         REAL(DBL_AD),INTENT(IN)::n

	   	 res%x_ad_ = n*u%x_ad_
		 res%xp_ad_= n*u%xp_ad_

	END FUNCTION MULT_RD

	!-----------------------------------------
	! MULTIPLY a dual number with REAL number
	! <n,0>*<u,up>=<u*n,up*n>
	!----------------------------------------
    ELEMENTAL FUNCTION MULT_SD(n,u) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u
		 TYPE (DUAL_NUM)::res
         REAL(SNG_AD),INTENT(IN)::n

	   	 res%x_ad_ = REAL(n,DBL_AD)*u%x_ad_
		 res%xp_ad_= REAL(n,DBL_AD)*u%xp_ad_

	END FUNCTION MULT_SD


!******* END: (*)
!---------------------


!******* BEGIN: (/)
!---------------------

    !-----------------------------------------
	! <u,up>/<v,vp>=<u/v,(up-u vp/v)/v>
	!----------------------------------------
    ELEMENTAL FUNCTION DIV_DD(u,v) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u,v
		 REAL(DBL_AD)::tmp
		 TYPE (DUAL_NUM)::res
         INTEGER:: i

		 tmp=1.D0/v%x_ad_
         res%x_ad_ = u%x_ad_*tmp
		 res%xp_ad_ =(u%xp_ad_- res%x_ad_*v%xp_ad_)*tmp

	END FUNCTION DIV_DD

    !-----------------------------------------
	! <u,up>/<n,0>=<u/n,up/n>
	!----------------------------------------
	ELEMENTAL FUNCTION DIV_DI(u,n) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u
		 INTEGER,INTENT(IN)::n
         REAL(DBL_AD)::tmp
    	 TYPE (DUAL_NUM)::res

		 tmp=1.D0/REAL(n,DBL_AD)
         res%x_ad_ = u%x_ad_*tmp
		 res%xp_ad_ =u%xp_ad_*tmp

	END FUNCTION DIV_DI

    !-----------------------------------------
	! DIVIDE dual number with respect to real numbers
	! <u,up>/<n,0>=<u/n,up/v>
	!----------------------------------------
	ELEMENTAL FUNCTION DIV_DR(u,n) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u
		 REAL(DBL_AD),INTENT(IN)::n
		 TYPE (DUAL_NUM):: res
         REAL(DBL_AD)::tmp

		 tmp=1.0D0/n
         res%x_ad_ = u%x_ad_*tmp
		 res%xp_ad_ =u%xp_ad_*tmp

	END FUNCTION DIV_DR


    !-----------------------------------------
	! DIVIDE dual number with respect to single
	! <u,up>/<n,0>=<u/n,up/v>
	!----------------------------------------
	ELEMENTAL FUNCTION DIV_DS(u,n) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u
		 REAL(SNG_AD),INTENT(IN)::n
		 TYPE (DUAL_NUM):: res
         REAL(DBL_AD)::tmp

		 tmp=1.0D0/REAL(n,DBL_AD)
         res%x_ad_ = u%x_ad_*tmp
		 res%xp_ad_ =u%xp_ad_*tmp

	END FUNCTION DIV_DS

	!-----------------------------------------
	! integer/<v,vp>
	! <n,0>/<v,vp>=<n/v,-n vp/v/v>=n*<1,-vp/v>/v
	!----------------------------------------
	ELEMENTAL FUNCTION DIV_ID(n,v) RESULT(res)
	     INTEGER,INTENT(IN)::n
	     TYPE (DUAL_NUM), INTENT(IN)::v
		 REAL(DBL_AD)::tmp,tmp2
		 TYPE (DUAL_NUM)::res

		 tmp=1.D0/v%x_ad_
         res%x_ad_=REAL(n,DBL_AD)*tmp
		 res%xp_ad_=-res%x_ad_*tmp*v%xp_ad_

	END FUNCTION DIV_ID


    !-----------------------------------------
	! real/<v,vp>
	! <n,0>/<v,vp>=<n/v,-n vp/v/v>=n*<1,-vp/v>/v
	!----------------------------------------
	ELEMENTAL FUNCTION DIV_RD(n,v) RESULT(res)
	     REAL(DBL_AD),INTENT(IN)::n
	     TYPE (DUAL_NUM), INTENT(IN)::v
		 REAL(DBL_AD)::tmp,tmp2
		 TYPE (DUAL_NUM)::res

		 tmp=1.D0/v%x_ad_
         res%x_ad_=n*tmp
		 res%xp_ad_=-res%x_ad_*tmp*v%xp_ad_

	END FUNCTION DIV_RD

    !-----------------------------------------
	! single/<v,vp>
	! <n,0>/<v,vp>=<n/v,-n vp/v/v>=n*<1,-vp/v>/v
	!----------------------------------------
	ELEMENTAL FUNCTION DIV_SD(n,v) RESULT(res)
	     REAL(SNG_AD),INTENT(IN)::n
	     TYPE (DUAL_NUM), INTENT(IN)::v
		 REAL(DBL_AD)::tmp,tmp2
		 TYPE (DUAL_NUM)::res

		 tmp=1.D0/v%x_ad_
         res%x_ad_=REAL(n,DBL_AD)*tmp
		 res%xp_ad_=-res%x_ad_*tmp*v%xp_ad_

	END FUNCTION DIV_SD

!******* END: (/)
!---------------------


!******* BEGIN: (**)
!---------------------

	!-----------------------------------------
	! POWER dual numbers
	! <u,up>^k=<u^k,k u^{k-1} up>
	!----------------------------------------
    ELEMENTAL FUNCTION POW_I(u,k) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
		 INTEGER,INTENT(IN)::k
		 REAL(DBL_AD)::tmp
		 TYPE (DUAL_NUM)::res

		 tmp=u%x_ad_**(k-1)
         res%x_ad_ = u%x_ad_*tmp
         res%xp_ad_=REAL(k,DBL_AD)*tmp*u%xp_ad_

	END FUNCTION POW_I

    !-----------------------------------------
	! POWER dual numbers
	! <u,up>^k=<u^k,k u^{k-1} up>
	!----------------------------------------
    ELEMENTAL FUNCTION POW_R(u,k) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
		 REAL(DBL_AD),INTENT(IN)::k
		 REAL(DBL_AD)::tmp
		 TYPE (DUAL_NUM)::res

		 tmp=u%x_ad_**(k-1)
         res%x_ad_ = u%x_ad_*tmp
         res%xp_ad_=k*tmp*u%xp_ad_

	END FUNCTION POW_R

    !-----------------------------------------
	! POWER dual numbers
	! <u,up>^k=<u^k,k u^{k-1} up>
	!----------------------------------------
    ELEMENTAL FUNCTION POW_S(u,k) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
		 REAL(SNG_AD),INTENT(IN)::k
		 REAL(DBL_AD)::tmp
		 TYPE (DUAL_NUM)::res

		 tmp=u%x_ad_**(k-1)
         res%x_ad_ = u%x_ad_*tmp
         res%xp_ad_=k*tmp*u%xp_ad_

	END FUNCTION POW_S

	!-----------------------------------------
	! POWER dual numbers to a dual power
	! <u,up>^(v,vp)=<u^v,u^v (v/u*up+Log(u)*vp>
	!----------------------------------------
    ELEMENTAL FUNCTION POW_D(u,v) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
		 TYPE (DUAL_NUM), INTENT(IN)::v
		 REAL(DBL_AD)::uf,vf
		 TYPE (DUAL_NUM)::res
		 uf=u%x_ad_
		 vf=v%x_ad_

		 res%x_ad_ =uf**vf
		 res%xp_ad_=res%x_ad_*(vf/uf*u%xp_ad_+LOG(uf)*v%xp_ad_)

	END FUNCTION POW_D
!******* END: (**)
!---------------------


!******* BEGIN: (==)
!---------------------
    !-----------------------------------------
	! compare two dual numbers, simply compare
	! the functional value.
	!----------------------------------------
    ELEMENTAL FUNCTION EQ_DD(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs,rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ == rhs%x_ad_)

	END FUNCTION EQ_DD

   ! compare a dual with an integer
   !-----------------------------------------
    ELEMENTAL FUNCTION EQ_DI(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     INTEGER,INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ == REAL(rhs,DBL_AD))

	END FUNCTION EQ_DI

    !-----------------------------------------
	! compare a dual number with a real number, simply compare
	! the functional value.
	!----------------------------------------
    ELEMENTAL FUNCTION EQ_DR(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     REAL(DBL_AD),INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ == rhs)

	END FUNCTION EQ_DR

   ! compare a dual with a single
   !-----------------------------------------
    ELEMENTAL FUNCTION EQ_DS(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     REAL(SNG_AD),INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ == REAL(rhs,DBL_AD))

	END FUNCTION EQ_DS

    !-----------------------------------------
	! compare an integer with a dual
	!----------------------------------------
    ELEMENTAL FUNCTION EQ_ID(lhs, rhs) RESULT(res)
		 INTEGER,INTENT(IN)::lhs
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (REAL(lhs,DBL_AD)==rhs%x_ad_)

	END FUNCTION EQ_ID

    !-----------------------------------------
	! compare a real with a dual
	!----------------------------------------
    ELEMENTAL FUNCTION EQ_RD(lhs, rhs) RESULT(res)
		 REAL(DBL_AD),INTENT(IN)::lhs
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (lhs==rhs%x_ad_)

    END FUNCTION EQ_RD

    !-----------------------------------------
	! compare a single with a dual
	!----------------------------------------
    ELEMENTAL FUNCTION EQ_SD(lhs, rhs) RESULT(res)
		 REAL(SNG_AD),INTENT(IN)::lhs
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (REAL(lhs,DBL_AD)==rhs%x_ad_)

    END FUNCTION EQ_SD

!******* END: (==)
!---------------------


!******* BEGIN: (<=)
!---------------------
    !-----------------------------------------
	! compare two dual numbers, simply compare
	! the functional value.
	!----------------------------------------
    ELEMENTAL FUNCTION LE_DD(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs,rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ <= rhs%x_ad_)

	END FUNCTION LE_DD

   ! compare a dual with an integer
   !-----------------------------------------
	ELEMENTAL FUNCTION LE_DI(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     INTEGER,INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ <= REAL(rhs,DBL_AD))

	END FUNCTION LE_DI

    !-----------------------------------------
	! compare a dual number with a real number, simply compare
	! the functional value.
	!----------------------------------------
    ELEMENTAL FUNCTION LE_DR(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     REAL(DBL_AD),INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ <= rhs)

	END FUNCTION LE_DR

  ! compare a dual with a single
   !----------------------------------------
	ELEMENTAL FUNCTION LE_DS(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     REAL(SNG_AD),INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ <= REAL(rhs,DBL_AD))

	END FUNCTION LE_DS

    !-----------------------------------------
	! compare a dual number with an integer
	!----------------------------------------
    ELEMENTAL FUNCTION LE_ID(n, rhs) RESULT(res)
		 INTEGER,INTENT(IN)::n
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (REAL(n,DBL_AD)<=rhs%x_ad_)

	END FUNCTION LE_ID

!-----------------------------------------
	! compare a real with a dual
	!----------------------------------------
    ELEMENTAL FUNCTION LE_RD(lhs, rhs) RESULT(res)
		 REAL(DBL_AD),INTENT(IN)::lhs
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (lhs<=rhs%x_ad_)

    END FUNCTION LE_RD

    !-----------------------------------------
	! compare a single with a dual
	!----------------------------------------
    ELEMENTAL FUNCTION LE_SD(lhs, rhs) RESULT(res)
		 REAL(SNG_AD),INTENT(IN)::lhs
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (REAL(lhs,DBL_AD)<=rhs%x_ad_)

    END FUNCTION LE_SD

!******* END: (<=)
!---------------------

!******* BEGIN: (<)
!---------------------
    !-----------------------------------------
	! compare two dual numbers, simply compare
	! the functional value.
	!----------------------------------------
    ELEMENTAL FUNCTION LT_DD(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs,rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ < rhs%x_ad_)

	END FUNCTION LT_DD

   ! compare a dual with an integer
   !-----------------------------------------
    ELEMENTAL FUNCTION LT_DI(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     INTEGER,INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ < REAL(rhs,DBL_AD))

	END FUNCTION LT_DI

    !-----------------------------------------
	! compare a dual number with a real number, simply compare
	! the functional value.
	!----------------------------------------
    ELEMENTAL FUNCTION LT_DR(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     REAL(DBL_AD),INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ < rhs)

	END FUNCTION LT_DR

   ! compare a dual with a single
   !----------------------------------------
    ELEMENTAL FUNCTION LT_DS(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     REAL(SNG_AD),INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ < REAL(rhs,DBL_AD))

	END FUNCTION LT_DS

 !-----------------------------------------
	! compare a dual number with an integer
	!----------------------------------------
    ELEMENTAL FUNCTION LT_ID(n, rhs) RESULT(res)
		 INTEGER,INTENT(IN)::n
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (REAL(n,DBL_AD)<rhs%x_ad_)

	END FUNCTION LT_ID

    !-----------------------------------------
	! compare a real with a dual
	!----------------------------------------
    ELEMENTAL FUNCTION LT_RD(lhs, rhs) RESULT(res)
		 REAL(DBL_AD),INTENT(IN)::lhs
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (lhs < rhs%x_ad_)

    END FUNCTION LT_RD

    !-----------------------------------------
	! compare a single with a dual
	!----------------------------------------
    ELEMENTAL FUNCTION LT_SD(lhs, rhs) RESULT(res)
		 REAL(SNG_AD),INTENT(IN)::lhs
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (REAL(lhs,DBL_AD) < rhs%x_ad_)

    END FUNCTION LT_SD


!******* END: (<)
!---------------------


!******* BEGIN: (>=)
!---------------------
    !-----------------------------------------
	! compare two dual numbers, simply compare
	! the functional value.
	!----------------------------------------
    ELEMENTAL FUNCTION GE_DD(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs,rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ >= rhs%x_ad_)

	END FUNCTION GE_DD

   ! compare a dual with an integer
   !-----------------------------------------
    ELEMENTAL FUNCTION GE_DI(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     INTEGER,INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ >= REAL(rhs,DBL_AD))

	END FUNCTION GE_DI

    !-----------------------------------------
	! compare a dual number with a real number, simply compare
	! the functional value.
	!----------------------------------------
    ELEMENTAL FUNCTION GE_DR(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     REAL(DBL_AD),INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ >= rhs)

	END FUNCTION GE_DR

   ! compare a dual with a single
   !----------------------------------------
    ELEMENTAL FUNCTION GE_DS(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     REAL(SNG_AD),INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ >=REAL(rhs,DBL_AD))

	END FUNCTION GE_DS
 !-----------------------------------------
	! compare a dual number with an integer
	!----------------------------------------
    ELEMENTAL FUNCTION GE_ID(n, rhs) RESULT(res)
		 INTEGER,INTENT(IN)::n
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (REAL(n,DBL_AD)>=rhs%x_ad_)

	END FUNCTION GE_ID

!-----------------------------------------
	! compare a real with a dual
	!----------------------------------------
    ELEMENTAL FUNCTION GE_RD(lhs, rhs) RESULT(res)
		 REAL(DBL_AD),INTENT(IN)::lhs
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (lhs>=rhs%x_ad_)

    END FUNCTION GE_RD

    !-----------------------------------------
	! compare a single with a dual
	!----------------------------------------
    ELEMENTAL FUNCTION GE_SD(lhs, rhs) RESULT(res)
		 REAL(SNG_AD),INTENT(IN)::lhs
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (REAL(lhs,DBL_AD)>=rhs%x_ad_)

    END FUNCTION GE_SD


!******* END: (>=)
!---------------------


!******* BEGIN: (>)
!---------------------
    !-----------------------------------------
	! compare two dual numbers, simply compare
	! the functional value.
	!----------------------------------------
    ELEMENTAL FUNCTION GT_DD(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs,rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ > rhs%x_ad_)

	END FUNCTION GT_DD

   ! compare a dual with an integer
   !-----------------------------------------
    ELEMENTAL FUNCTION GT_DI(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     INTEGER,INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ > REAL(rhs,DBL_AD))

	END FUNCTION GT_DI

    !-----------------------------------------
	! compare a dual number with a real number, simply compare
	! the functional value.
	!----------------------------------------
    ELEMENTAL FUNCTION GT_DR(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     REAL(DBL_AD),INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ > rhs)

	END FUNCTION GT_DR

   ! compare a dual with a single
   !----------------------------------------
    ELEMENTAL FUNCTION GT_DS(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     REAL(SNG_AD),INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ >REAL(rhs,DBL_AD))

	END FUNCTION GT_DS

  !-----------------------------------------
	! compare a dual number with an integer
	!----------------------------------------
    ELEMENTAL FUNCTION GT_ID(n, rhs) RESULT(res)
		 INTEGER,INTENT(IN)::n
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (REAL(n,DBL_AD)>rhs%x_ad_)

	END FUNCTION GT_ID

!-----------------------------------------
	! compare a real with a dual
	!----------------------------------------
    ELEMENTAL FUNCTION GT_RD(lhs, rhs) RESULT(res)
		 REAL(DBL_AD),INTENT(IN)::lhs
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (lhs>rhs%x_ad_)

    END FUNCTION GT_RD

    !-----------------------------------------
	! compare a single with a dual
	!----------------------------------------
    ELEMENTAL FUNCTION GT_SD(lhs, rhs) RESULT(res)
		 REAL(SNG_AD),INTENT(IN)::lhs
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (REAL(lhs,DBL_AD)>rhs%x_ad_)

    END FUNCTION GT_SD


!******* END: (>)
!---------------------



!******* BEGIN: (/=)
!---------------------
    !-----------------------------------------
	! compare two dual numbers, simply compare
	! the functional value.
	!----------------------------------------
    ELEMENTAL FUNCTION NE_DD(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs,rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ /= rhs%x_ad_)

	END FUNCTION NE_DD

   ! compare a dual with an integer
   !-----------------------------------------
    ELEMENTAL FUNCTION NE_DI(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     INTEGER,INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ /= REAL(rhs,DBL_AD))

	END FUNCTION NE_DI

    !-----------------------------------------
	! compare a dual number with a real number, simply compare
	! the functional value.
	!----------------------------------------
    ELEMENTAL FUNCTION NE_DR(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     REAL(DBL_AD),INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ /= rhs)

	END FUNCTION NE_DR

   ! compare a dual with a single
   !----------------------------------------
    ELEMENTAL FUNCTION NE_DS(lhs, rhs) RESULT(res)
		 TYPE (DUAL_NUM), INTENT(IN):: lhs
	     REAL(SNG_AD),INTENT(IN)::rhs
		 LOGICAL::res

	     res = (lhs%x_ad_ /=REAL(rhs,DBL_AD))

	END FUNCTION NE_DS

 !-----------------------------------------
	! compare a dual number with an integer
	!----------------------------------------
    ELEMENTAL FUNCTION NE_ID(n, rhs) RESULT(res)
		 INTEGER,INTENT(IN)::n
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (REAL(n,DBL_AD)/=rhs%x_ad_)

	END FUNCTION NE_ID

!-----------------------------------------
	! compare a real with a dual
	!----------------------------------------
    ELEMENTAL FUNCTION NE_RD(lhs, rhs) RESULT(res)
		 REAL(DBL_AD),INTENT(IN)::lhs
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (lhs/=rhs%x_ad_)

    END FUNCTION NE_RD

    !-----------------------------------------
	! compare a single with a dual
	!----------------------------------------
    ELEMENTAL FUNCTION NE_SD(lhs, rhs) RESULT(res)
		 REAL(SNG_AD),INTENT(IN)::lhs
		 TYPE (DUAL_NUM), INTENT(IN):: rhs
		 LOGICAL::res

	     res = (REAL(lhs,DBL_AD)/=rhs%x_ad_)

    END FUNCTION NE_SD

!******* END: (/=)
!---------------------

	!---------------------------------------------------
	! ABS of dual numbers
	! ABS<u,up>=<ABS(u),up SIGN(u)>
	!----------------------------------------------------
	ELEMENTAL FUNCTION ABS_D(u) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
         TYPE (DUAL_NUM)::res

         IF(u%x_ad_>0) THEN
		    res%x_ad_ = u%x_ad_
			res%xp_ad_= u%xp_ad_
         ELSE IF (u%x_ad_<0) THEN
		    res%x_ad_ = -u%x_ad_
	        res%xp_ad_= -u%xp_ad_
		 ELSE
            res%x_ad_ = 0.0D0
	        res%xp_ad_= 0.0 !Set_NaN() ! Indicating an undefined derivative, however, for some codes it will cause problem.
		 ENDIF

	END FUNCTION ABS_D


	!-----------------------------------------
	! ACOS of dual numbers
	! ACOS<u,up>=<ACOS(u),-up/sqrt(1-(u%x_ad_)**2)>
	!----------------------------------------
	ELEMENTAL FUNCTION ACOS_D(u) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
         TYPE (DUAL_NUM)::res
         REAL(DBL_AD)::tmp

         res%x_ad_ = ACOS(u%x_ad_)
		 IF(u%x_ad_==1.0D0.OR.u%x_ad_==-1.0D0) THEN
			res%xp_ad_=Set_NaN() ! Indicating an undefined derivative
		 ELSE
			tmp= -1.0d0/SQRT(1.0D0-(u%x_ad_)**2)
			res%xp_ad_= u%xp_ad_*tmp
         ENDIF
	END FUNCTION ACOS_D


    !-----------------------------------------
	! ASIN of dual numbers
	! ASIN<u,up>=<ASIN(u),up 1/SQRT(1-u^2)>
	!----------------------------------------
	ELEMENTAL FUNCTION ASIN_D(u) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
         TYPE (DUAL_NUM)::res
         REAL (DBL_AD):: tmp

         res%x_ad_ = ASIN(u%x_ad_)
		 IF(u%x_ad_==1.0D0.OR.u%x_ad_==-1.0D0) THEN
			res%xp_ad_=Set_NaN() ! Indicating an undefined derivative
		 ELSE
			tmp= 1.0d0/SQRT(1.0D0-(u%x_ad_)**2)
			res%xp_ad_= u%xp_ad_*tmp
         ENDIF
	END FUNCTION ASIN_D


  	!-----------------------------------------
	! COS of dual numbers
	! COS<u,up>=<COS(u),-up*SIN(u)>
	!----------------------------------------
	ELEMENTAL FUNCTION COS_D(u) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
         TYPE (DUAL_NUM)::res
		 REAL(DBL_AD):: tmp

         res%x_ad_ = COS(u%x_ad_)
		 tmp=-SIN(u%x_ad_)
         res%xp_ad_= u%xp_ad_*tmp

	END FUNCTION COS_D

    !-----------------------------------------
	! DOT PRODUCT two dual number vectors
	! <u,up>.<v,vp>=<u.v,u.vp+up.v>
	!----------------------------------------
	FUNCTION DOT_PRODUCT_DD(u,v) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u(:),v(:)
		 TYPE (DUAL_NUM)::res
         INTEGER:: i

         res%x_ad_ = DOT_PRODUCT(u%x_ad_,v%x_ad_)

		 DO i=1,NDV_AD
			res%xp_ad_(i) =DOT_PRODUCT(u%x_ad_,v%xp_ad_(i))+DOT_PRODUCT(u%xp_ad_(i),v%x_ad_)
	     ENDDO
	END FUNCTION DOT_PRODUCT_DD



    !-----------------------------------------
	! EXPONENTIAL OF dual numbers
	! EXP<u,up>=<EXP(u),up EXP(u)>
	!----------------------------------------
	ELEMENTAL FUNCTION EXP_D(u) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u
		 REAL(DBL_AD)::tmp
		 TYPE (DUAL_NUM)::res

		 tmp=EXP(u%x_ad_)
		 res%x_ad_ = tmp
		 res%xp_ad_ =u%xp_ad_*tmp

	END FUNCTION EXP_D

    !-----------------------------------------
	! EXPONENTIAL OF dual numbers
	! EXP<u,up>=<EXP(u),up EXP(u)>
	!----------------------------------------
	ELEMENTAL FUNCTION INT_D(u) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u
		 REAL(DBL_AD)::tmp
		 INTEGER::res

		 tmp=u%x_ad_
		 res = INT(tmp)

	END FUNCTION INT_D


    !-----------------------------------------
	! LOG OF dual numbers,defined for u%x>0 only
	! the error control should be done in the original code
	! in other words, if u%x<=0, it is not possible to obtain LOG.
	! LOG<u,up>=<LOG(u),up/u>
	!----------------------------------------
	ELEMENTAL FUNCTION LOG_D(u) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u
		 TYPE (DUAL_NUM)::res
		 REAL(DBL_AD)::tmp

		 res%x_ad_ = LOG(u%x_ad_)
		 tmp=1.0d0/u%x_ad_
		 res%xp_ad_ =u%xp_ad_*tmp

	END FUNCTION LOG_D

!-----------------------------------------
	! LOG OF dual numbers,defined for u%x>0 only
	! the error control should be done in the original code
	! in other words, if u%x<=0, it is not possible to obtain LOG.
	! LOG<u,up>=<LOG(u),up/u>
	!----------------------------------------
	ELEMENTAL FUNCTION LOG10_D(u) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u
		 TYPE (DUAL_NUM)::res
		 REAL(DBL_AD)::tmp

		 res%x_ad_ = LOG10(u%x_ad_)
		 tmp=1.0d0/u%x_ad_/LOG(10.0D0)
		 res%xp_ad_ =u%xp_ad_*tmp

	END FUNCTION LOG10_D


    !-----------------------------------------
	! MULTIPLY two dual number matrices
	!
	! <u,up>.<v,vp>=<u.v,up.v+u.vp>
	!----------------------------------------
	FUNCTION MATMUL_DD(u,v) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u(:,:),v(:,:)
         TYPE (DUAL_NUM)::res(SIZE(u,1),SIZE(v,2))
		 INTEGER:: i

         res%x_ad_ = MATMUL(u%x_ad_,v%x_ad_)
		 DO i=1,NDV_AD
			res%xp_ad_(i)= MATMUL(u%xp_ad_(i),v%x_ad_) + MATMUL(u%x_ad_,v%xp_ad_(i))
         ENDDO
	END FUNCTION MATMUL_DD

    !-----------------------------------------
	! MULTIPLY a dual number matrix with a dual number
	! vector
	!
	! <u,up>.<v,vp>=<u.v,up.v+u.vp>
	!----------------------------------------
	FUNCTION MATMUL_DV(u,v) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u(:,:),v(:)
         TYPE (DUAL_NUM)::res(SIZE(u,1))
         INTEGER:: i

         res%x_ad_ = MATMUL(u%x_ad_,v%x_ad_)
		 DO i=1,NDV_AD
			res%xp_ad_(i)= MATMUL(u%xp_ad_(i),v%x_ad_) + MATMUL(u%x_ad_,v%xp_ad_(i))
         ENDDO
	END FUNCTION MATMUL_DV


    !-----------------------------------------
	! MULTIPLY a dual vector with a  dual matrix
	!
	! <u,up>.<v,vp>=<u.v,up.v+u.vp>
	!----------------------------------------
	FUNCTION MATMUL_VD(u,v) RESULT(res)
	     TYPE (DUAL_NUM), INTENT(IN)::u(:),v(:,:)
         TYPE (DUAL_NUM)::res(SIZE(v,2))
         INTEGER::i

         res%x_ad_= MATMUL(u%x_ad_,v%x_ad_)
		 DO i=1,NDV_AD
			res%xp_ad_(i)= MATMUL(u%xp_ad_(i),v%x_ad_) + MATMUL(u%x_ad_,v%xp_ad_(i))
         ENDDO
	END FUNCTION MATMUL_VD

    !-----------------------------------------
	! Obtain the max of 2 to 5 dual numbers
	!----------------------------------------
	ELEMENTAL FUNCTION MAX_DD(val1, val2, val3, val4,val5) RESULT(res)
		TYPE (DUAL_NUM), INTENT(IN)::val1, val2
		TYPE (DUAL_NUM), INTENT(IN),OPTIONAL:: val3, val4,val5
		TYPE (DUAL_NUM)::res

		IF (val1%x_ad_ > val2%x_ad_) THEN
			res = val1
		ELSE
			res = val2
		ENDIF
        IF(PRESENT(val3))THEN
		   IF(res%x_ad_<val3%x_ad_) res=val3
        ENDIF
        IF(PRESENT(val4))THEN
		   IF(res%x_ad_<val4%x_ad_) res=val4
        ENDIF
		IF(PRESENT(val5))THEN
		   IF(res%x_ad_<val5%x_ad_) res=val5
        ENDIF

    END FUNCTION MAX_DD


    !-----------------------------------------
	! Obtain the max of a dual number and an integer
	!----------------------------------------
	ELEMENTAL FUNCTION MAX_DI(u, n) RESULT(res)
		TYPE (DUAL_NUM), INTENT(IN)::u
		INTEGER,INTENT(IN)::n
		TYPE (DUAL_NUM)::res

		IF (u%x_ad_ > n) THEN
			res = u
		ELSE
			res = n
		ENDIF

    END FUNCTION MAX_DI

    !-----------------------------------------
	! Obtain the max of a dual number and a real number
	!----------------------------------------
	ELEMENTAL FUNCTION MAX_DR(u, n) RESULT(res)
		TYPE (DUAL_NUM), INTENT(IN)::u
		REAL(DBL_AD),INTENT(IN)::n
		TYPE (DUAL_NUM)::res

		IF (u%x_ad_ > n) THEN
			res = u
		ELSE
			res = n
		ENDIF

    END FUNCTION MAX_DR


 !-----------------------------------------
	! Obtain the max of a dual number and a single
	!----------------------------------------
	ELEMENTAL FUNCTION MAX_DS(u, n) RESULT(res)
		TYPE (DUAL_NUM), INTENT(IN)::u
		REAL(SNG_AD),INTENT(IN)::n
		TYPE (DUAL_NUM)::res

		IF (u%x_ad_ > n) THEN
			res = u
		ELSE
			res = n
		ENDIF

    END FUNCTION MAX_DS


    !---------------------------------------------------
	! Obtain the max of a real and a dual
	! note the real argument is renamed as r to avoid
	! the ambiguity due to positional association using
	! keywords when functions/subroutines are called
	!---------------------------------------------------
	 ELEMENTAL FUNCTION MAX_RD(r,u) RESULT(res)
		REAL(DBL_AD),INTENT(IN)::r
		TYPE (DUAL_NUM), INTENT(IN)::u
		TYPE (DUAL_NUM)::res

		IF (u%x_ad_ > r) THEN
			res = u
		ELSE
			res = r
		ENDIF

    END FUNCTION MAX_RD

    !-----------------------------------------
	! Obtain the max value of vector u
	!----------------------------------------
	FUNCTION MAXVAL_D(u) RESULT(res)
		TYPE (DUAL_NUM), INTENT(IN)::u(:)
		INTEGER::iloc(1)
		TYPE (DUAL_NUM)::res

        iloc=MAXLOC(u%x_ad_)
		res=u(iloc(1))

    END FUNCTION MAXVAL_D


    !-----------------------------------------
	! Obtain the min of 2 to 4 dual numbers
	!----------------------------------------
    ELEMENTAL FUNCTION MIN_DD(val1, val2, val3, val4) RESULT(res)
		TYPE (DUAL_NUM), INTENT(IN)::val1, val2
		TYPE (DUAL_NUM), INTENT(IN),OPTIONAL:: val3, val4
		TYPE (DUAL_NUM)::res

		IF (val1%x_ad_ < val2%x_ad_) THEN
			res = val1
		ELSE
			res = val2
		ENDIF
        IF(PRESENT(val3))THEN
		   IF(res%x_ad_>val3%x_ad_) res=val3
        ENDIF
        IF(PRESENT(val4))THEN
		   IF(res%x_ad_>val4%x_ad_) res=val4
        ENDIF

    END FUNCTION MIN_DD

!-----------------------------------------
	! Obtain the min of a dual number and a single
	!----------------------------------------
	ELEMENTAL FUNCTION MIN_DR(u, n) RESULT(res)
		TYPE (DUAL_NUM), INTENT(IN)::u
		REAL(DBL_AD),INTENT(IN)::n
		TYPE (DUAL_NUM)::res

		IF (u%x_ad_ < n) THEN
			res = u
		ELSE
			res = n
		ENDIF

    END FUNCTION MIN_DR

    !-----------------------------------------
	! Obtain the min of a dual number and a single
	!----------------------------------------
	ELEMENTAL FUNCTION MIN_DS(u, n) RESULT(res)
		TYPE (DUAL_NUM), INTENT(IN)::u
		REAL(SNG_AD),INTENT(IN)::n
		TYPE (DUAL_NUM)::res

		IF (u%x_ad_ < n) THEN
			res = u
		ELSE
			res = n
		ENDIF

    END FUNCTION MIN_DS

  !-----------------------------------------
	! Obtain the min value of vector u
	!----------------------------------------
	FUNCTION MINVAL_D(u) RESULT(res)
		TYPE (DUAL_NUM), INTENT(IN)::u(:)
		INTEGER::iloc(1)
		TYPE (DUAL_NUM)::res

        iloc=MINLOC(u%x_ad_)
		res=u(iloc(1))

    END FUNCTION MINVAL_D

    !------------------------------------------------------
    !Returns the nearest integer to u%x, ELEMENTAL
    !------------------------------------------------------
	ELEMENTAL FUNCTION NINT_D(u) RESULT(res)
		TYPE (DUAL_NUM), INTENT(IN):: u
		INTEGER::res

		res=NINT(u%x_ad_)

    END FUNCTION NINT_D



    !----------------------------------------------------------------
	! SIGN(a,b) with two dual numbers as inputs,
	! the result will be |a| if b%x>=0, -|a| if b%x<0,ELEMENTAL
    !----------------------------------------------------------------
	ELEMENTAL   FUNCTION SIGN_DD(val1, val2) RESULT(res)
		TYPE (DUAL_NUM), INTENT(IN) :: val1, val2
		TYPE (DUAL_NUM)::res

		IF (val2%x_ad_ < 0.D0) THEN
			res = -ABS(val1)
		ELSE
			res =  ABS(val1)
		ENDIF

     END FUNCTION SIGN_DD


    !----------------------------------------------------------------
	! SIGN(a,b) with one real and one dual number as inputs,
	! the result will be |a| if b%x>=0, -|a| if b%x<0,ELEMENTAL
    !----------------------------------------------------------------
	ELEMENTAL   FUNCTION SIGN_RD(val1, val2) RESULT(res)
		REAL(DBL_AD),INTENT(IN)::val1
		TYPE (DUAL_NUM), INTENT(IN) :: val2
		TYPE (DUAL_NUM)::res

		IF (val2%x_ad_ < 0.D0) THEN
			res = -ABS(val1)
		ELSE
			res = ABS(val1)
		ENDIF

     END FUNCTION SIGN_RD


    !-----------------------------------------
	! SIN of dual numbers
	! SIN<u,up>=<SIN(u),up COS(u)>
	!----------------------------------------
	ELEMENTAL FUNCTION SIN_D(u) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
         TYPE (DUAL_NUM)::res
         REAL (DBL_AD):: tmp

         res%x_ad_ = SIN(u%x_ad_)
		 tmp=COS(u%x_ad_)
         res%xp_ad_= u%xp_ad_*tmp

	END FUNCTION SIN_D


    !-----------------------------------------
	! SQRT of dual numbers
	! SQRT<u,up>=<SQRT(u),up/2/sqrt(u) >
	!----------------------------------------
	ELEMENTAL FUNCTION SQRT_D(u) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u
		 REAL(DBL_AD):: tmp
		 TYPE (DUAL_NUM)::res

		 tmp=SQRT(u%x_ad_)

         res%x_ad_ = tmp
		 IF(tmp==0.0d0) THEN
            res%xp_ad_=Set_NaN()
		 ELSE
			tmp= 0.5D0/tmp
			res%xp_ad_= u%xp_ad_*tmp
         ENDIF
	END FUNCTION SQRT_D



    !-----------------------------------------
	! SUM OF A DUAL ARRAY
    !----------------------------------------
	FUNCTION SUM_D(u) RESULT(res)
         TYPE (DUAL_NUM), INTENT(IN)::u(:)
		 TYPE (DUAL_NUM)::res
		 INTEGER:: i

         res%x_ad_ = SUM(u%x_ad_)
		 DO i=1,NDV_AD
			res%xp_ad_(i)= SUM(u%xp_ad_(i))
         ENDDO
	END FUNCTION SUM_D

ELEMENTAL FUNCTION Set_NaN() RESULT(res)
		 REAL(DBL_AD)::res

         res=SQRT(negative_one)

	END FUNCTION Set_NaN



END MODULE  Dual_Num_Auto_Diff


