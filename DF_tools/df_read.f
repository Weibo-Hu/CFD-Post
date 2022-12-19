      program read_inflow

      ! NOTE: Compile with ifort -r8 -o df_gridgen_dimensions.x df_gridgen_dimensions.f90
      !
      ! INFO: This program generates the input data necessary for the "new" DF-technique (March2012).
      !       The input can be taken from any datafile by specifying the correct columns in the NAMELIST
      !       Note: only the columns on the "right side" of the y column are interpolated
      !             so take care that y is the first column in the source file or the columns on the
      !             "left side" of y are unimportant for the DF-technique (e.g. x, p, etc...)

      implicit none

      integer                          :: ioout = 3000
      integer                          :: i1, i2, n1, n2
      real                             :: a1, a2, a3, a4, a5, a6, a7
      real                             :: a8, a9, a10, a11, a12, a13
!     character(len=100)               :: pathout = '.'
      real, allocatable                :: p1(:), p2(:)
      real                             :: t1(202), t2(402)
      real                             :: var0(2), var1(5)                            ! target variable written in boundary_input_0001.inp
      real                             :: var2(6)                            ! target variable written in boundary_input_0001.inp
      logical                          :: file_found                    



      n1 = 200
      n2 = 400
      allocate( p1(0:n1+1))
      allocate( p2(0:n2+1))
      inquire( file='boundary_input_000001.inp', exist = file_found )
      if( file_found )then
          print*, "input file found"
          open( ioout , file='boundary_input_000001.inp' , 
     &          form='unformatted' )  ! This file (binary) is needed for the DF                                                                                           ! technique
      else
          print*, "input file not found"
      end if

      do i2 = 0 , n2 + 1
        do i1 = 0 , n1 + 1
!          read( ioout ) a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13
           ! read( ioout ) t1(i1), t2(i2), var1, var2
           read( ioout ) a1, a2, var1, var2
           
           ! read( ioout ) a1, a2
           ! t1(i1) = var0(1)
           ! t2(i2) = var0(2)
         end do
      end do
      print*, "y= ", a1
      print*, "z= ", a2
      close( ioout )

      deallocate( p1 , p2 )

      end program read_inflow