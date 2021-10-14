      real function spmpar(i)
      integer i
c     **********
c
c     Function spmpar
c
c     This function provides single precision machine parameters
c     when the appropriate set of data statements is activated (by
c     removing the c from column 1) and all other data statements are
c     rendered inactive. Most of the parameter values were obtained
c     from the corresponding Bell Laboratories Port Library function.
c
c     The function statement is
c
c       real function spmpar(i)
c
c     where
c
c       i is an integer input variable set to 1, 2, or 3 which
c         selects the desired machine parameter. If the machine has
c         t base b digits and its smallest and largest exponents are
c         emin and emax, respectively, then these parameters are
c
c         spmpar(1) = b**(1 - t), the machine precision,
c
c         spmpar(2) = b**(emin - 1), the smallest magnitude,
c
c         spmpar(3) = b**emax*(1 - b**(-t)), the largest magnitude.
c
c     Argonne National Laboratory. MINPACK Project. November 1996.
c     Burton S. Garbow, Kenneth E. Hillstrom, Jorge J. More'
c
c     **********
      integer mcheps(2)
      integer minmag(2)
      integer maxmag(2)
      real rmach(3)
      equivalence (rmach(1),mcheps(1))
      equivalence (rmach(2),minmag(1))
      equivalence (rmach(3),maxmag(1))
c
c     Machine constants for the IBM 360/370 series,
c     the Amdahl 470/V6, the ICL 2900, the Itel AS/6,
c     the Xerox Sigma 5/7/9 and the Sel systems 85/86.
c
c     data rmach(1) / z3c100000 /
c     data rmach(2) / z00100000 /
c     data rmach(3) / z7fffffff /
c
c     Machine constants for the Honeywell 600/6000 series.
c
c     data rmach(1) / o716400000000 /
c     data rmach(2) / o402400000000 /
c     data rmach(3) / o376777777777 /
c
c     Machine constants for the CDC 6000/7000 series.
c
c     data rmach(1) / 16414000000000000000b /
c     data rmach(2) / 00014000000000000000b /
c     data rmach(3) / 37767777777777777777b /
c
c     Machine constants for the PDP-10 (KA or KI processor).
c
c     data rmach(1) / "147400000000 /
c     data rmach(2) / "000400000000 /
c     data rmach(3) / "377777777777 /
c
c     Machine constants for the PDP-11 fortran supporting
c     32-bit integers (expressed in integer and octal).
c
c     data mcheps(1) /  889192448 /
c     data minmag(1) /    8388608 /
c     data maxmag(1) / 2147483647 /
c
c     data rmach(1) / o06500000000 /
c     data rmach(2) / o00040000000 /
c     data rmach(3) / o17777777777 /
c
c     Machine constants for the PDP-11 fortran supporting
c     16-bit integers (expressed in integer and octal).
c
c     data mcheps(1),mcheps(2) / 13568,     0 /
c     data minmag(1),minmag(2) /   128,     0 /
c     data maxmag(1),maxmag(2) / 32767,    -1 /
c
c     data mcheps(1),mcheps(2) / o032400, o000000 /
c     data minmag(1),minmag(2) / o000200, o000000 /
c     data maxmag(1),maxmag(2) / o077777, o177777 /
c
c     Machine constants for the Burroughs 5700/6700/7700 systems.
c
c     data rmach(1) / o1301000000000000 /
c     data rmach(2) / o1771000000000000 /
c     data rmach(3) / o0777777777777777 /
c
c     Machine constants for the Burroughs 1700 system.
c
c     data rmach(1) / z4ea800000 /
c     data rmach(2) / z400800000 /
c     data rmach(3) / z5ffffffff /
c
c     Machine constants for the Univac 1100 series.
c
c     data rmach(1) / o147400000000 /
c     data rmach(2) / o000400000000 /
c     data rmach(3) / o377777777777 /
c
c     Machine constants for the Data General Eclipse S/200.
c
c     Note - it may be appropriate to include the following card -
c     static rmach(3)
c
c     data minmag/20k,0/,maxmag/77777k,177777k/
c     data mcheps/36020k,0/
c
c     Machine constants for the Harris 220.
c
c     data mcheps(1) / '20000000, '00000353 /
c     data minmag(1) / '20000000, '00000201 /
c     data maxmag(1) / '37777777, '00000177 /
c
c     Machine constants for the Cray-1.
c
c     data rmach(1) / 0377224000000000000000b /
c     data rmach(2) / 0200034000000000000000b /
c     data rmach(3) / 0577777777777777777776b /
c
c     Machine constants for the Prime 400.
c
c     data mcheps(1) / :10000000153 /
c     data minmag(1) / :10000000000 /
c     data maxmag(1) / :17777777777 /
c
c     Machine constants for the VAX-11.
c
c     data mcheps(1) /  13568 /
c     data minmag(1) /    128 /
c     data maxmag(1) / -32769 /
c
c     Machine constants for IEEE machines.
c
      data rmach(1) /1.192091E-07/
      data rmach(2) /1.175495E-38/
      data rmach(3) /3.402822E+38/
c
      spmpar = rmach(i)
      return
c
c     Last card of function spmpar.
c
      end