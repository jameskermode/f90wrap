      subroutine sgeco(a,lda,n,ipvt,rcond,z)
      integer lda,n,ipvt(1)
      real a(lda,1),z(1)
      real rcond
c
c     sgeco factors a real matrix by gaussian elimination
c     and estimates the condition of the matrix.
c
c     if  rcond  is not needed, sgefa is slightly faster.
c     to solve  a*x = b , follow sgeco by sgesl.
c     to compute  inverse(a)*c , follow sgeco by sgesl.
c     to compute  determinant(a) , follow sgeco by sgedi.
c     to compute  inverse(a) , follow sgeco by sgedi.
c
c     on entry
c
c        a       real(lda, n)
c                the matrix to be factored.
c
c        lda     integer
c                the leading dimension of the array  a .
c
c        n       integer
c                the order of the matrix  a .
c
c     on return
c
c        a       an upper triangular matrix and the multipliers
c                which were used to obtain it.
c                the factorization can be written  a = l*u  where
c                l  is a product of permutation and unit lower
c                triangular matrices and  u  is upper triangular.
c
c        ipvt    integer(n)
c                an integer vector of pivot indices.
c
c        rcond   real
c                an estimate of the reciprocal condition of  a .
c                for the system  a*x = b , relative perturbations
c                in  a  and  b  of size  epsilon  may cause
c                relative perturbations in  x  of size  epsilon/rcond .
c                if  rcond  is so small that the logical expression
c                           1.0 + rcond .eq. 1.0
c                is true, then  a  may be singular to working
c                precision.  in particular,  rcond  is zero  if
c                exact singularity is detected or the estimate
c                underflows.
c
c        z       real(n)
c                a work vector whose contents are usually unimportant.
c                if  a  is close to a singular matrix, then  z  is
c                an approximate null vector in the sense that
c                norm(a*z) = rcond*norm(a)*norm(z) .
c
c     linpack. this version dated 08/14/78 .
c     cleve moler, university of new mexico, argonne national lab.
c
c     subroutines and functions
c
c     linpack sgefa
c     blas saxpy,sdot,sscal,sasum
c     fortran abs,amax1,sign
c
c     internal variables
c
      real sdot,ek,t,wk,wkm
      real anorm,s,sasum,sm,ynorm
      integer info,j,k,kb,kp1,l
c
c
c     compute 1-norm of a
c
      anorm = 0.0e0
      do 10 j = 1, n
         anorm = amax1(anorm,sasum(n,a(1,j),1))
   10 continue
c
c     factor
c
      call sgefa(a,lda,n,ipvt,info)
c
c     rcond = 1/(norm(a)*(estimate of norm(inverse(a)))) .
c     estimate = norm(z)/norm(y) where  a*z = y  and  trans(a)*y = e .
c     trans(a)  is the transpose of a .  the components of  e  are
c     chosen to cause maximum local growth in the elements of w  where
c     trans(u)*w = e .  the vectors are frequently rescaled to avoid
c     overflow.
c
c     solve trans(u)*w = e
c
      ek = 1.0e0
      do 20 j = 1, n
         z(j) = 0.0e0
   20 continue
      do 100 k = 1, n
         if (z(k) .ne. 0.0e0) ek = sign(ek,-z(k))
         if (abs(ek-z(k)) .le. abs(a(k,k))) go to 30
            s = abs(a(k,k))/abs(ek-z(k))
            call sscal(n,s,z,1)
            ek = s*ek
   30    continue
         wk = ek - z(k)
         wkm = -ek - z(k)
         s = abs(wk)
         sm = abs(wkm)
         if (a(k,k) .eq. 0.0e0) go to 40
            wk = wk/a(k,k)
            wkm = wkm/a(k,k)
         go to 50
   40    continue
            wk = 1.0e0
            wkm = 1.0e0
   50    continue
         kp1 = k + 1
         if (kp1 .gt. n) go to 90
            do 60 j = kp1, n
               sm = sm + abs(z(j)+wkm*a(k,j))
               z(j) = z(j) + wk*a(k,j)
               s = s + abs(z(j))
   60       continue
            if (s .ge. sm) go to 80
               t = wkm - wk
               wk = wkm
               do 70 j = kp1, n
                  z(j) = z(j) + t*a(k,j)
   70          continue
   80       continue
   90    continue
         z(k) = wk
  100 continue
      s = 1.0e0/sasum(n,z,1)
      call sscal(n,s,z,1)
c
c     solve trans(l)*y = w
c
      do 120 kb = 1, n
         k = n + 1 - kb
         if (k .lt. n) z(k) = z(k) + sdot(n-k,a(k+1,k),1,z(k+1),1)
         if (abs(z(k)) .le. 1.0e0) go to 110
            s = 1.0e0/abs(z(k))
            call sscal(n,s,z,1)
  110    continue
         l = ipvt(k)
         t = z(l)
         z(l) = z(k)
         z(k) = t
  120 continue
      s = 1.0e0/sasum(n,z,1)
      call sscal(n,s,z,1)
c
      ynorm = 1.0e0
c
c     solve l*v = y
c
      do 140 k = 1, n
         l = ipvt(k)
         t = z(l)
         z(l) = z(k)
         z(k) = t
         if (k .lt. n) call saxpy(n-k,t,a(k+1,k),1,z(k+1),1)
         if (abs(z(k)) .le. 1.0e0) go to 130
            s = 1.0e0/abs(z(k))
            call sscal(n,s,z,1)
            ynorm = s*ynorm
  130    continue
  140 continue
      s = 1.0e0/sasum(n,z,1)
      call sscal(n,s,z,1)
      ynorm = s*ynorm
c
c     solve  u*z = v
c
      do 160 kb = 1, n
         k = n + 1 - kb
         if (abs(z(k)) .le. abs(a(k,k))) go to 150
            s = abs(a(k,k))/abs(z(k))
            call sscal(n,s,z,1)
            ynorm = s*ynorm
  150    continue
         if (a(k,k) .ne. 0.0e0) z(k) = z(k)/a(k,k)
         if (a(k,k) .eq. 0.0e0) z(k) = 1.0e0
         t = -z(k)
         call saxpy(k-1,t,a(1,k),1,z(1),1)
  160 continue
c     make znorm = 1.0
      s = 1.0e0/sasum(n,z,1)
      call sscal(n,s,z,1)
      ynorm = s*ynorm
c
      if (anorm .ne. 0.0e0) rcond = ynorm/anorm
      if (anorm .eq. 0.0e0) rcond = 0.0e0
      return
      end