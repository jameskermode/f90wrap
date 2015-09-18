!******************************************************************************
!* Dual Number Automatic Differentiation (DNAD) of Fortran Codes
!* -------------------------------------------------------------
!* COPYRIGHT (c) Wenbin Yu, All rights reserved, you are free to copy, 
!* modify or translate this code to other languages such as c/c++. If 
!* you find a bug please let me know through wenbinyu.heaven@gmail.com. If 
!* you added new functions and want to share with others, please let me know too.
!* You are welcome to share your successful stories with us through 
!* http://groups.google.com/group/hifi-comp. 
!******************************************************************************
!* Simple Instruction:
!*---------------------
!* This is the header file: define the interface needed for overloading intrinsic 
!* functions and operators.  This file should put in the same folder as dnad.f90. 
!* If a function or operation is not defined (unlikely), you need to create a 
!* corresponding interface in this file (DNADHeaders.f90) and a corresponding 
!* function/subroutine in DNAD.f90.
!*********************************************************************************
!* Acknowledgements
!----------------------
!* The development of DNAD is supported, in part, by the Chief Scientist 
!* Innovative Research Fund at AFRL/RB WPAFB, and by Department of Army 
!* SBIR (Topic A08-022) through Advanced Dynamics Inc. The views and 
!* conclusions contained herein are those of the authors and should not be 
!* interpreted as necessarily representing the official policies or endorsement,
!* either expressed or implied, of the funding agency.    
!*********************************************************************************

!******** Interfaces for operator overloading
    PUBLIC ASSIGNMENT (=)
    INTERFACE ASSIGNMENT (=)
		MODULE PROCEDURE ASSIGN_DI  ! dual=integer, ELEMENTAL
		MODULE PROCEDURE ASSIGN_DR  ! dual=real, ELEMENTAL
		MODULE PROCEDURE ASSIGN_DS  ! dual=integer, ELEMENTAL
		MODULE PROCEDURE ASSIGN_ID  ! integer=dual, ELEMENTAL

!It is found out that compilers will know how to assign a scalar to vectors and matrices 
!if the assignment is defined for assigning a scalar to a dual number. Hence it
!is unnecessary to define such assignment overloadings. 
     	    	
	END INTERFACE

	
	PUBLIC OPERATOR (+)
	INTERFACE OPERATOR (+)	
		MODULE PROCEDURE ADD_D   ! +dual number, ELEMENTAL
		MODULE PROCEDURE ADD_DD  ! dual+ dual, ELEMENTAL
		MODULE PROCEDURE ADD_DI  ! dual+ integer, ELEMENTAL
	    MODULE PROCEDURE ADD_DR  ! dual+ real, ELEMENTAL
        MODULE PROCEDURE ADD_DS  ! dual+ single, ELEMENTAL
		MODULE PROCEDURE ADD_ID  ! integer+dual, ELEMENTAL
		MODULE PROCEDURE ADD_RD  ! real+ dual, ELEMENTAL
	    MODULE PROCEDURE ADD_SD  ! single+dual, ELEMENTAL
!It is found out that these overloads also cover the cases when one of the operand is a matrix or vector.
!Of course, if both operands are vectors or matrices, they should be of the same shape 
	END INTERFACE

	PUBLIC OPERATOR (-)
	INTERFACE OPERATOR (-)
		MODULE PROCEDURE MINUS_D   ! negate a dual number,ELEMENTAL	
		MODULE PROCEDURE MINUS_DD  ! dual -dual,ELEMENTAL
		MODULE PROCEDURE MINUS_DI  ! dual-integer,ELEMENTAL
		MODULE PROCEDURE MINUS_DR  ! dual-real,ELEMENTAL
		MODULE PROCEDURE MINUS_DS  ! dual-single,ELEMENTAL
		MODULE PROCEDURE MINUS_ID  ! integer-dual,ELEMENTAL
		MODULE PROCEDURE MINUS_RD  ! real-dual,ELEMENTAL
		MODULE PROCEDURE MINUS_SD  ! single-dual,ELEMENTAL
!It is found out that these overloads also cover the cases when one of the operand is a matrix or vector.
!Of course, if both operands are vectors or matrices, they should be of the same shape 
	END INTERFACE

	PUBLIC OPERATOR (*)
	INTERFACE OPERATOR (*)
		MODULE PROCEDURE MULT_DD    ! dual*dual, ELEMENTAL
		MODULE PROCEDURE MULT_DI    ! dual*integer,ELEMENTAL
		MODULE PROCEDURE MULT_DR    ! dual*real,ELEMENTAL
		MODULE PROCEDURE MULT_DS    ! dual*single,ELEMENTAL
		MODULE PROCEDURE MULT_ID    ! integer*dual,ELEMENTAL
		MODULE PROCEDURE MULT_RD    ! real*dual,ELEMENTAL
		MODULE PROCEDURE MULT_SD    ! single*dual,ELEMENTAL
!It is found out that these overloads also cover the cases when one of the operand is a matrix or vector.
!Of course, if both operands are vectors or matrices, they should be of the same shape 
	END INTERFACE

	PUBLIC OPERATOR (/)
	INTERFACE OPERATOR (/)
		MODULE PROCEDURE DIV_DD ! dual/dual,ELEMENTAL
		MODULE PROCEDURE DIV_DI ! dual/integer, ELEMENTAL
		MODULE PROCEDURE DIV_DR ! dual/real,EMENTAL
		MODULE PROCEDURE DIV_DS ! dual/single,EMENTAL
		MODULE PROCEDURE DIV_ID ! integer/dual, ELEMENTAL
		MODULE PROCEDURE DIV_RD ! real/dual, ELEMENTAL
		MODULE PROCEDURE DIV_SD ! single/dual, ELEMENTAL	
	END INTERFACE

    PUBLIC OPERATOR (**)
	INTERFACE OPERATOR (**)
		MODULE PROCEDURE POW_I ! power a dual number to an integer power,ELEMENTAL
		MODULE PROCEDURE POW_R ! power a dual number to a real (double precision) power, ELEMENTAL
		MODULE PROCEDURE POW_S ! power a dual number to a real (single precision) power, ELEMENTAL
		MODULE PROCEDURE POW_D ! power a dual number to a dual power, ELEMENTAL
	END INTERFACE
   
    PUBLIC OPERATOR (==)
	INTERFACE OPERATOR (==)
		MODULE PROCEDURE EQ_DD ! compare two dual numbers, ELEMENTAL
		MODULE PROCEDURE EQ_DI ! compare a dual and an integer, ELEMENTAL
		MODULE PROCEDURE EQ_DR ! compare a dual and a real, ELEMENTAL
		MODULE PROCEDURE EQ_DS ! compare a dual and a single, ELEMENTAL
        MODULE PROCEDURE EQ_ID ! compare an integer with a dual number, ELEMENTAL
        MODULE PROCEDURE EQ_RD ! compare a real with a dual number, ELEMENTAL
        MODULE PROCEDURE EQ_SD ! compare a single with a dual number, ELEMENTAL
	END INTERFACE
    
	PUBLIC OPERATOR (<=)
	INTERFACE OPERATOR (<=)
		MODULE PROCEDURE LE_DD  ! compare two dual numbers, ELEMENTAL
		MODULE PROCEDURE LE_DI  ! compare a dual and an integer, ELEMENTAL
		MODULE PROCEDURE LE_DR  ! compare a dual and a real,ELEMENTAL 
		MODULE PROCEDURE LE_DS  ! compare a dual and a single,ELEMENTAL 
		MODULE PROCEDURE LE_ID ! compare an integer with a dual number, ELEMENTAL
        MODULE PROCEDURE LE_RD ! compare a real with a dual number, ELEMENTAL
        MODULE PROCEDURE LE_SD ! compare a single with a dual number, ELEMENTAL
	END INTERFACE
    
	PUBLIC OPERATOR (<)
	INTERFACE OPERATOR (<)
		MODULE PROCEDURE LT_DD  !compare two dual numbers, ELEMENTAL
		MODULE PROCEDURE LT_DI  !compare a dual and an integer, ELEMENTAL
		MODULE PROCEDURE LT_DR  !compare dual with a real, ELEMENTAL
		MODULE PROCEDURE LT_DS ! compare <u,up> and a single
		MODULE PROCEDURE LT_ID ! compare an integer with a dual number, ELEMENTAL
        MODULE PROCEDURE LT_RD ! compare a real with a dual number, ELEMENTAL
        MODULE PROCEDURE LT_SD ! compare a single with a dual number, ELEMENTAL
	END INTERFACE

    PUBLIC OPERATOR (>=)
	INTERFACE OPERATOR (>=)
		MODULE PROCEDURE GE_DD ! compare two dual numbers, ELEMENTAL
		MODULE PROCEDURE GE_DI ! compare dual with integer, ELEMENTAL
		MODULE PROCEDURE GE_DR ! compare a dual number with a real number, ELEMENTAL
        MODULE PROCEDURE GE_DS ! compare dual with a single, ELEMENTAL
		MODULE PROCEDURE GE_ID ! compare an integer with a dual number, ELEMENTAL
        MODULE PROCEDURE GE_RD ! compare a real with a dual number, ELEMENTAL
        MODULE PROCEDURE GE_SD ! compare a single with a dual number, ELEMENTAL
	END INTERFACE

    PUBLIC OPERATOR (>)
	INTERFACE OPERATOR (>)
	    MODULE PROCEDURE GT_DD  !compare two dual numbers, ELEMENTAL
		MODULE PROCEDURE GT_DI  !compare a dual and an integer, ELEMENTAL
		MODULE PROCEDURE GT_DR  !compare dual with a real, ELEMENTAL
		MODULE PROCEDURE GT_DS ! compare <u,up> and a single
    	MODULE PROCEDURE GT_ID ! compare an integer with a dual number, ELEMENTAL
        MODULE PROCEDURE GT_RD ! compare a real with a dual number, ELEMENTAL
        MODULE PROCEDURE GT_SD ! compare a single with a dual number, ELEMENTAL
	END INTERFACE
  
    PUBLIC OPERATOR (/=)
	INTERFACE OPERATOR (/=)
	    MODULE PROCEDURE NE_DD  !compare two dual numbers, ELEMENTAL
		MODULE PROCEDURE NE_DI  !compare a dual and an integer, ELEMENTAL
		MODULE PROCEDURE NE_DR  !compare dual with a real, ELEMENTAL
		MODULE PROCEDURE NE_DS ! compare <u,up> and a single
		MODULE PROCEDURE NE_ID ! compare an integer with a dual number, ELEMENTAL
        MODULE PROCEDURE NE_RD ! compare a real with a dual number, ELEMENTAL
        MODULE PROCEDURE NE_SD ! compare a single with a dual number, ELEMENTAL
	END INTERFACE
	
	
!------------------------------------------------
! Interfaces for intrinsic functions overloading
!------------------------------------------------
   PUBLIC ABS
   INTERFACE ABS
 	   MODULE PROCEDURE ABS_D  ! obtain the absolute value of a dual number, ELEMENTAL
   END INTERFACE
 
   PUBLIC DABS
   INTERFACE DABS
 	   MODULE PROCEDURE ABS_D ! the same as ABS, used for some old fortran commands
   END INTERFACE

   PUBLIC ACOS
   INTERFACE ACOS
 	   MODULE PROCEDURE ACOS_D ! obtain the arccosine of a dual number, ELEMENTAL
   END INTERFACE
 
  PUBLIC ASIN
   INTERFACE ASIN
 	   MODULE PROCEDURE ASIN_D ! obtain the arcsine of a dual number, ELEMENTAL
   END INTERFACE
 
   PUBLIC COS
   INTERFACE COS
 	   MODULE PROCEDURE COS_D ! obtain the cosine of a dual number, ELEMENTAL
   END INTERFACE
 
   PUBLIC DCOS
   INTERFACE DCOS
 	   MODULE PROCEDURE COS_D ! obtain the cosine of a dual number, ELEMENTAL
   END INTERFACE
  
   
   PUBLIC DOT_PRODUCT
   INTERFACE DOT_PRODUCT
 	   MODULE PROCEDURE DOT_PRODUCT_DD ! dot product two dual number vectors
   END INTERFACE

   PUBLIC EXP 
   INTERFACE EXP
 	   MODULE PROCEDURE EXP_D ! obtain the exponential of a dual number, ELEMENTAL
   END INTERFACE


   PUBLIC INT 
   INTERFACE INT
 	   MODULE PROCEDURE INT_D ! obtain the integer part of a dual number, ELEMENTAL
   END INTERFACE

   PUBLIC LOG 
   INTERFACE LOG
 	   MODULE PROCEDURE LOG_D ! obtain the log of a dual number, ELEMENTAL
   END INTERFACE
   
   PUBLIC LOG10 
   INTERFACE LOG10
 	   MODULE PROCEDURE LOG10_D ! obtain the log of a dual number, ELEMENTAL
   END INTERFACE
 
   PUBLIC MATMUL
   INTERFACE MATMUL
 	   MODULE PROCEDURE MATMUL_DD ! matrix multiplies of two dual matrices
 	   MODULE PROCEDURE MATMUL_DV ! matrix multiplies of a dual matrix with a dual vector
 	   MODULE PROCEDURE MATMUL_VD ! matrix multiplies of a dual vector with a dual matrix
   END INTERFACE
   

   PUBLIC MAX
   INTERFACE MAX
 	   MODULE PROCEDURE MAX_DD ! obtain the max of from two to four dual numbers, ELEMENTAL
	   MODULE PROCEDURE MAX_DI ! obtain the max of from a dual number and an integer, ELEMENTAL
	   MODULE PROCEDURE MAX_DR ! obtain the max of from a dual number and a real, ELEMENTAL
	   MODULE PROCEDURE MAX_DS ! obtain the max of from a dual number and a real in single precision, ELEMENTAL
	   MODULE PROCEDURE MAX_RD ! obtain the max of from a real,and a dual number,  ELEMENTAL
   END INTERFACE

   PUBLIC DMAX1
   INTERFACE DMAX1
 	   MODULE PROCEDURE MAX_DD ! obtain the max of from two to four dual numbers, ELEMENTAL
   END INTERFACE
   
   PUBLIC MAXVAL
   INTERFACE MAXVAL
 	   MODULE PROCEDURE MAXVAL_D ! obtain the maxval  of a dual number vectgor
   END INTERFACE
   
   PUBLIC MIN
   INTERFACE MIN
 	   MODULE PROCEDURE MIN_DD ! obtain the min of from two to four dual numbers, ELEMENTAL
  	   MODULE PROCEDURE MIN_DR ! obtain the min of a dual and a real, ELEMENTAL
  	   MODULE PROCEDURE MIN_DS ! obtain the min of a dual and a single, ELEMENTAL
   END INTERFACE

   PUBLIC DMIN1
   INTERFACE DMIN1
 	   MODULE PROCEDURE MIN_DD ! obtain the min of from two to four dual numbers, ELEMENTAL
   END INTERFACE
 
   PUBLIC MINVAL
   INTERFACE MINVAL
 	   MODULE PROCEDURE MINVAL_D ! obtain the maxval  of a dual number vectgor
   END INTERFACE
  
   PUBLIC NINT
   INTERFACE NINT
 	   MODULE PROCEDURE NINT_D ! Returns the nearest integer to the argument, ELEMENTAL
   END INTERFACE

   PUBLIC SIGN   
   INTERFACE  SIGN
     MODULE PROCEDURE  SIGN_DD ! SIGN(a,b) with two dual numbers as inputs, the result will be |a| if b%x>=0, -|a| if b%x<0,ELEMENTAL
     MODULE PROCEDURE  SIGN_RD ! SIGN(a,b) with a real and a dual, the result will be |a| if b%x>=0, -|a| if b%x<0,ELEMENTAL
   END INTERFACE

   PUBLIC SIN  
   INTERFACE SIN
 	   MODULE PROCEDURE SIN_D ! obtain sine of a dual number, ELEMENTAL
   END INTERFACE

   PUBLIC DSIN  
   INTERFACE DSIN
 	   MODULE PROCEDURE SIN_D ! obtain sine of a dual number, ELEMENTAL
   END INTERFACE

   PUBLIC SQRT  
   INTERFACE SQRT
 	   MODULE PROCEDURE SQRT_D ! obtain the sqrt of a dual number, ELEMENTAL
   END INTERFACE

   PUBLIC SUM  
   INTERFACE SUM
 	   MODULE PROCEDURE SUM_D ! sum a dual array
   END INTERFACE

