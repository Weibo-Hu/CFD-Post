      program inflow_field

      ! NOTE: Compile with ifort -r8 -o df_input_profile.x df_input_profile.f90
      !
      ! INFO: This program generates the input data necessary for the "new" DF-technique (March2012).
      !       The input can be taken from any datafile by specifying the correct columns in the NAMELIST
      !       Note: only the columns on the "right side" of the y column are interpolated
      !             so take care that y is the first column in the source file or the columns on the
      !             "left side" of y are unimportant for the DF-technique (e.g. x, p, etc...)

      implicit none

!---- source parameters
      integer                          :: nvinp                                  ! (NAMELIST) number of variables in source file
      integer                          :: nlinp                                  ! (NAMELIST) number of lines (data points) source file
      integer                          :: nskip, nskip_grid                      ! (NAMELIST) 4, 32 lines to skip in czpgttblx_xxxxx.dat file (tecplot header) (from python output based on temporal les)
      integer, dimension(13)           :: column = 0.                            ! (NAMELIST) y, uVD, u, v, w, rho, T, R11, R22, R33, R12
                                                                                 !            NOTE: only the columns on the "right side" of the y column are interpolated

      character(len=100)               :: input_file_name

!---- Target grid parameters
      integer                          :: n1, n2                                 ! (NAMELIST) grid point target grid
      integer                          :: n1tot, n2tot                           ! np + buffer
      integer, parameter               :: buffer = 1

      real                             :: l10 , l11                              ! (NAMELIST) domain start/end in 1 direction (y in my case)
      real                             :: l20 , l21                              ! (NAMELIST) domain start/end in 2 direction (z in my case)
      real                             :: p1 , p2                                ! (NAMELIST) bunching law parameters
      real, allocatable                :: g1(:), g2(:)

      real                             :: u_damp, v_damp, w_damp                 ! reference damp velocities
      real                             :: rho_damp, T_damp                       ! reference damp density / temperature

      integer  , parameter             :: nv = 11                                ! number of variables written in boundary_input_0001.inp (necessary for df_routine)
      real     , allocatable, Target   :: inp(:,:)
      real     , allocatable, target   :: var (:,:,:)                            ! calculations are perfomed with this variable
      real     , allocatable           :: tvar(:,:,:)                            ! target variable written in boundary_input_0001.inp
      real     , pointer               :: y_s(:), u_s(:), uVD_s(:)               ! s means source file specified in namelist (value haven't been interpolated)
      real                             :: damplength                          &  ! random number quantifying the "speed of damping"
                                        , damping_start                          ! defines where damping starts, measured in boundary layer thicknesses

      integer                          :: i, j, l, i1, i2, jdelta0

      integer                          :: ioinp = 10
      integer                          :: ioout = 11
      integer                          :: ioerr

      real , allocatable               :: e(:,:), p(:,:)
      real , parameter                 :: gamma = 1.4
      real                             :: Mach                                  ! (NAMELIST) free stream Mach number
      real                             :: ekin
      real                             :: u_ref = 601.                        & ! Reference values within TTBL simulation
                                        , rho_ref = 0.558138                  &
                                        , l_ref   = 0.001073                  &
                                        , T_ref = 100.                        &
                                        , y_wall = 0.                         &
                                        , R_univ = 8314.4621                  & ! j/kmolK
                                        , M_Air  = 28.96512                   &
                                        , p_ref, p_soll, cp, cv

      real, parameter                  :: small=1.d-30
      real                             :: uVD_delta, delta_VD, h, delta0
      real                             :: delta1_VD, dy_1, I_x(3), x
      integer                          :: nmy, n_zone1, idelta

      logical                          :: file_found                          &
                                        , damping = .false.                   &  ! (NAMELIST) damp v component to zero
                                        , read_target_grid
      character(len=400)               :: pathin_grid, pathout           ! (NAMELIST) source and target
      character( len=6 )               :: type1, type2                           ! (NAMELIST) line bunching type
      character(len=400)               :: inputfile = 'config_df_input_profile.inp'    ! name of the input configuration file

      real, allocatable                :: t1(:), t2(:)
      real                             :: var1(5), var2(6)                            ! target variable written in boundary_input_0001.inp

      namelist /CONFIGURATION/    Mach, u_ref, &
                                  rho_ref, l_ref, T_ref, &
                                  M_Air, &
                                  n1, n2, &
                                  l10, l11, &
                                  l20, l21, &
                                  type1, type2, &
                                  p1, p2, &
                                  pathin_grid, &
                                  pathout, nlinp, &
                                  nvinp, nskip, nskip_grid, &
                                  column, damping, &
                                  damplength, damping_start, &
                                  read_target_grid, &
                                  u_damp, v_damp, w_damp, &
                                  rho_damp, T_damp, &
                                  input_file_name


      inquire( file=inputfile, exist = file_found )
      if( file_found )then
          open( unit = ioinp , file = inputfile , status = 'old' )
      else
          call ABORT("input file not found")
      end if

      read(ioinp,nml=CONFIGURATION, iostat = ioerr)
      if( ioerr /= 0 )then
          print *, "error code: ", ioerr
          call ABORT('Cannot read CONFIGURATION from '//inputfile)
      end if
      close(ioinp)

      ! Order in boundary_input.inp: <y>  <u>  <v>  <w>  <rho>  <T>  <uu>  <vv>  <ww>  <uv>

      allocate( inp(nlinp,nvinp) )


      n1tot = n1 + 2 * buffer  ! target: NO. of grid points in direction-1
      n2tot = n2 + 2 * buffer  ! target: NO. of grid points in direction-2

      allocate( g1(1:n1tot) )  ! coordinates in direction-1
      allocate( g2(1:n2tot) )  ! coordinates in direction-2
      allocate( var (1:n1tot,1:n2tot,nvinp ) )  ! input : store all variables in 2D mesh
      allocate( tvar(1:n1tot,1:n2tot,nv    ) )  ! target: store all variabels in 2D mesh

      ! The current case not read_target_grid
      if(.not. read_target_grid)then
!--- generate direction-1, y

      call gridgen( n1 , buffer , g1 , p1 , type1 , l10 , l11 )

!--- generate direction-2, z

      call gridgen( n2 , buffer , g2 , p2 , type2 , l20 , l21 )
      end if

      ! Skip this step
      if(read_target_grid)then
         g1(1)     = l10
         g1(n1tot) = l11
         open( ioinp , file=pathin_grid , form="formatted" )
         do j = 1,nskip_grid
            read( ioinp, * )
         end do
         do j = 1, n1
            read( ioinp , * ) g1(j+1)
         end do
         call gridgen( n2 , buffer , g2 , p2 , type2 , l20 , l21 )
         close( ioinp )
      end if

!--- generate target field

      open( ioinp, file = input_file_name, form='formatted' )
      ! read initial turbulence data, NO of lines * NO of variables
      do j = 1, nlinp
         read(ioinp,*) inp(j,:)
      end do
      close( ioinp )
      ! interpolate input profile(all variables) into target grid in direction-1
      print*, "l_ref=", l_ref
      print*, "g1=", g1(1)
      do i = 1, n1tot
         var(i,:,1) = g1(i)
         do j = 2, nvinp
!           x = g1(i)-g1(1) / l_ref
            x = g1(i) / l_ref
            var(i,:,j) = INTERPOL(nlinp,inp(:,column(1)),inp(:,j),x)
         end do
      end do

      if(column( 3) > 0 )then
         tvar(:,:,1 ) = var(:,:,column( 3)) ! --- <u>
      else
         tvar(:,:,1 ) = 0.
      end if

      if( column( 4) > 0 )then
         tvar(:,:,2 ) = var(:,:,column( 4)) ! --- <v>
      else
         tvar(:,:,2 ) = 0.
      end if
      if( column( 5) > 0 )then
         tvar(:,:,3 ) = var(:,:,column( 5)) ! --- <w>
      else
         tvar(:,:,3 ) = 0.
      end if
      if( column( 6) > 0 )then
         tvar(:,:,4 ) = var(:,:,column( 6)) ! --- <rho>
      else
         tvar(:,:,4 ) = 0.
      end if
      if( column( 7) > 0 )then
         tvar(:,:,5 ) = var(:,:,column( 7)) ! --- <T>
      else
         tvar(:,:,5 ) = 0.
      end if
      if( column( 8) > 0 )then
         tvar(:,:,6 ) = var(:,:,column( 8)) ! --- <R11>
      else
         tvar(:,:,6 ) = 0.
      end if
      if( column( 9) > 0 )then
         tvar(:,:,7 ) = var(:,:,column( 9)) ! --- <R22>
      else
         tvar(:,:,7 ) = 0.
      end if
      if( column(10) > 0 )then
         tvar(:,:,8 ) = var(:,:,column(10)) ! --- <R33>
      else
         tvar(:,:,8 ) = 0.
      end if
      if( column(11) > 0 )then
         tvar(:,:,9 ) = var(:,:,column(11)) ! --- <R21>
      else
         tvar(:,:,9 ) = 0.
      end if
      if( column(12) > 0 )then
         tvar(:,:,10) = var(:,:,column(12)) ! --- <R31>
      else
         tvar(:,:,10) = 0.
      end if
      if( column(13) > 0 )then
         tvar(:,:,11) = var(:,:,column(13)) ! --- <R32>
      else
         tvar(:,:,11) = 0.
      end if


!     tvar(:,:,1 ) = u_ref      ! --- <u>
!     tvar(:,:,2 ) = 0.         ! --- <v>
!     tvar(:,:,3 ) = 0.         ! --- <w>
!     tvar(:,:,4 ) = rho_ref    ! --- <rho>
!     tvar(:,:,5 ) = T_ref      ! --- <T>
!     tvar(:,:,6 ) = 0.16       ! --- <R11>
!     tvar(:,:,7 ) = 0.09       ! --- <R22>
!     tvar(:,:,8 ) = 0.04       ! --- <R33>
!     tvar(:,:,9 ) = 0.         ! --- <R21>
!     tvar(:,:,10) = 0.         ! --- <R31>
!     tvar(:,:,11) = 0.         ! --- <R32>

      cp = gamma / (gamma - 1) * R_univ/M_Air  ! 8314.4621/33640.3136 ??? or 8314.4624/29 ???
      cv = cp - R_univ/M_Air

!---- damp fluctuations to zero outside the boundary layer
      if( damping )then

         do j = 1 , n1 + 2*buffer
            tvar(j,:,6 ) = tvar(j,:,6 )                                    &
                         * exp( -( max(g1(j)-damping_start*delta0,0.)/damplength )**2 )
            tvar(j,:,7 ) = tvar(j,:,7 )                                    &
                         * exp( -( max(g1(j)-damping_start*delta0,0.)/damplength )**2 )
            tvar(j,:,8 ) = tvar(j,:,8 )                                    &
                         * exp( -( max(g1(j)-damping_start*delta0,0.)/damplength )**2 )
            tvar(j,:,9 ) = tvar(j,:,9 )                                    &
                         * exp( -( max(g1(j)-damping_start*delta0,0.)/damplength )**2 )
         end do

!---- damp mean flow properties to reference state
         do j = 1 , n1 + 2*buffer
             if( g1(j) .gt. damping_start*delta0 )then
               ! - u --> u_damp
               tvar(j,:,1 ) = u_damp + ( tvar(j,:,1 ) - u_damp )   &
                            * exp( -( max(g1(j)-damping_start*delta0,0.)/damplength )**2 )

               ! - v --> v_damp
               tvar(j,:,2 ) = v_damp + ( tvar(j,:,2 ) - v_damp )   &
                            * exp( -( max(g1(j)-damping_start*delta0,0.)/damplength )**2 )

               ! - w --> w_damp
               tvar(j,:,3 ) = w_damp + ( tvar(j,:,3 ) - w_damp )   &
                            * exp( -( max(g1(j)-damping_start*delta0,0.)/damplength )**2 )

               ! - rho --> rho_damp
               tvar(j,:,4 ) = rho_damp + ( tvar(j,:,4 ) - rho_damp )   &
                            * exp( -( max(g1(j)-damping_start*delta0,0.)/damplength )**2)

               ! - T --> T_damp
               tvar(j,:,5 ) = T_damp + ( tvar(j,:,5 ) - T_damp )   &
                            * exp( -( max(g1(j)-damping_start*delta0,0.)/damplength )**2 )
               end if
         end do


      end if



!--- compute energy
      allocate( e(1:n1tot,1:n2tot),p(1:n1tot,1:n2tot) )
      do j = 1 , n2 + 2 * buffer
        do i = 1 , n1 + 2 * buffer
          ekin   = 0.5 * tvar(i,j,4 ) * ( tvar(i,j,1)**2. + tvar(i,j,2)**2. + tvar(i,j,3)**2. )
          p(i,j) = tvar(i,j,4) * R_univ/M_Air * tvar(i,j,5)
          e(i,j) = tvar(i,j,4)*cv*tvar(i,j,5) + ekin                        ! rho*E_t
        end do
      end do

      open( ioout , file=trim(pathout)//'/boundary_input_000001.inp' &
          , form='unformatted' )  ! This file (binary) is needed for the DF
!                                                                                           ! technique
      print*, "n1tot=", n1 + 2 * buffer
      print*, "n2tot=", n2 + 2 * buffer
      do i2 = 1 , n2 + 2 * buffer
        do i1 = 1 , n1 + 2 * buffer
!      do i1 = 1 , n1 + 2 * buffer
!        do i2 = 1, n2 + 2 * buffer
           write( ioout ) g1(i1)        &
                        , g2(i2)        &
                        , tvar(i1,i2,:)
        end do
      end do
      close( ioout ) 

      open( ioout , file=trim(pathout)//'/boundary_input_000001.dat' , form='formatted' )  ! This file (ascii) is needed for the DF
                                                                                             ! technique
      do i2 = 1 , n2 + 2 * buffer
        do i1 = 1 , n1 + 2 * buffer
!      do i1 = 1 , n1 + 2 * buffer
!        do i2 = 1, n2 + 2 * buffer
           write( ioout , '(13(E18.8, 2X))' ) g1(i1) &
                        , g2(i2)        &
                        , tvar(i1,i2,:)
        end do
      end do
      close( ioout ) 

      open( ioout , file=trim(pathout)//'/boundary_input.inp' , form='formatted' )  ! serves for inca_incon.f (initialization of the flow
      write( ioout, * ) '"y", "z", "u", "v", "w", "rho", "E"' &
                  , ', "u`u`", "v`v`", "w`w`", "u`v`", "u`w`", "v`w`"'
      do i1 = 1 , n1 + 2 * buffer                                                   ! field with the mean values, however, not used in
                                                                                    ! my SOTON_UK_RANS case, since it is initialized with
                                                                                    ! the RANS mean values!
        write( ioout , '(13(ES18.8,2x))' ) g1(i1), 0. , tvar(i1,1,1), tvar(i1,1,2), tvar(i1,1,3), tvar(i1,1,4), e(i1,1) &
                     , 0. , 0. , 0. , 0. , 0. , 0.
      end do
      close( ioout )

      open( ioout , file=trim(pathout)//'/fluctuations.inp' , form='formatted' )    ! serves for inca_incon.f (initialization of the flow
      write( ioout, * ) '"y", "u`u`", "v`v`", "w`w`", "u`v`"'
      do i1 = 1 , n1 + 2 * buffer                                                   ! field with the mean values, however, not used in
                                                                                    ! my SOTON_UK_RANS case, since it is initialized with
                                                                                    ! the RANS mean values!
        write( ioout , '(5(ES18.8,2x))' ) g1(i1), tvar(i1,1,6), tvar(i1,1,7), tvar(i1,1,8), tvar(i1,1,9)
      end do
      close( ioout )

      call WRITE_OUTPUT( n1 , n2 , buffer  &
                       , nv                &
                       , g1 , g2           &
                       , pathout           &
                       , tvar, e, p )

      deallocate( inp )
      deallocate( e )
      deallocate( var , tvar )
      deallocate( g1 , g2 )

      contains

      subroutine GRIDGEN( nx , buf, gx , p1 , gtype &
                        , l0 , l1 )

      integer, intent(in) :: nx
      integer             :: npx
      integer, intent(in) :: buf
      real                :: gx(1:nx+2*buf)
      real                :: p1
      character(len=*)    :: gtype
      real   , intent(in) :: l0, l1

      real, allocatable   :: hx(:), hgx(:), dx(:)

      real                :: h0
      integer             :: i

      npx = nx + 2*buffer

      allocate( hx( 1 : npx ) , hgx( 1 : npx ) , dx( 1 : npx ) )

      select case( gtype )

      case('HOMO')

        dx(:) = ( l1-l0 ) / real(nx)
        hx(:) = ( l1-l0 ) / real(nx)

        gx(buf+1) = l0 + 0.5 * dx(buf)

        do i = buf , 1, -1
          gx(i) = gx(i+1) - dx(i)
        end do
        do i = buf+1 , npx-1
          gx(i+1) = gx(i) + dx(i)
        end do

        gx(1)   = l0
        gx(npx) = l1

      case('HYPTCF')

        h0 = 0.5 * ( l1-l0 )
        do i = 0 , nx
          hgx(i+buf) = -h0 * TANH( p1 *(1.-real(2*i)/real(nx)) ) / TANH( p1 )
        end do

        do i = 1 , nx
          hx (i+buf) = hgx(i+buf) - hgx(i+buf-1)
        end do

        hx( :buf ) = hx( buf+1 )
        hx( nx+buf+1: ) = hx( nx+buf )

        gx(buf+1) = l0 + 0.5 * hx(buf+1)
        do i = buf , 1, -1
          gx(i) = gx(i+1) - 0.5 * ( hx(i) + hx(i+1) )
        end do
        do i = buf+1 , npx-1
          gx(i+1) = gx(i) + 0.5 * ( hx(i) + hx(i+1) )
        end do

        do i = 1 , npx - 1
          dx(i) = gx(i+1) - gx(i)
        end do
        dx(npx) = 2.* dx(npx-1) - dx(npx-2)

        gx(1)   = l0
        gx(npx) = l1

      case('HYPTBL')

        h0 = l1-l0

        do i = 0 , nx
          hgx(i+buf) = h0 * ( 1. - TANH( p1*(1.-real(i)/real(nx)) )/ TANH( p1 ) )
        end do

        do i = buf + 1 , buf + nx
          hx(i) = hgx(i) - hgx(i-1)
        end do

        do i = 1 , buf
      !old  hx( buf+1 -i ) = hx( buf+1  )
      !old  hx( nx+buf+i ) = hx( nx+buf )
          hx( buf+1 -i ) = hx( buf+i  )           !Eric
          hx( nx+buf+i ) = hx( nx+buf+1-i )       !Eric
        end do

        gx(buf+1) = l0 + 0.5 * hx(buf+1)
        do i = buf , 1, -1
          gx(i) = gx(i+1) - 0.5 * ( hx(i) + hx(i+1) )
        end do
        do i = buf+1 , npx-1
          gx(i+1) = gx(i) + 0.5 * ( hx(i) + hx(i+1) )
        end do

        do i = 1 , npx - 1
          dx(i) = gx(i+1) - gx(i)
        end do

        dx(npx) = 2.* dx(npx-1) - dx(npx-2)

        gx(1)   = l0
        gx(npx) = l1

      case('HYPSIN')

         h0 = l1-l0

         do i = nx , 0 , -1

           hgx(nx - i + buffer) = h0 * ( SINH( p1 * ( 1. - real(i) / real(nx) ) ) &
                                / SINH(p1) )

         end do

         do i = buffer + 1 , buffer + nx
           hx(i) = hgx(i) - hgx(i-1)
         end do

         do i = 1 , buffer
      !old hx( buffer+1 -i ) = hx( buffer+1  )
      !old hx( nx+buffer+i ) = hx( nx+buffer )
           hx( buffer+1 -i ) = hx( buffer+i  )           !Eric
           hx( nx+buffer+i ) = hx( nx+buffer+1-i )       !Eric
         end do

         gx(buffer+1) = l0 + 0.5 * hx(buffer+1)

         do i = buffer , 1, -1
           gx(i) = gx(i+1) - 0.5 * ( hx(i) + hx(i+1) )
         end do

         do i = buffer+1 , npx-1
           gx(i+1) = gx(i) + 0.5 * ( hx(i) + hx(i+1) )
         end do

         do i = 1 , npx - 1
           dx(i) = gx(i+1) - gx(i)
         end do

         dx(npx) = 2.* dx(npx-1) - dx(npx-2)

         gx(1)   = l0
         gx(npx) = l1

      end select

      deallocate( hx , dx , hgx )

      end subroutine GRIDGEN

      function INTERPOL(nmx,pos,var,x)  result(y)

      implicit none

      integer                       :: nmx
      real                          :: pos (nmx) , var(nmx)
      real                          :: x , y
      integer                       :: n


      if (x .ge. pos(nmx)) then
        y = var(nmx)
      elseif (x .le. pos(1)) then
        y = var(1)
      else
        do n=1,nmx-1
          if (x.ge.pos(n).and.x.lt.pos(n+1)) then
            y = var(n) + (var(n+1)-var(n)) * (x - pos(n))/(pos(n+1)-pos(n))
            exit

          endif
        enddo

      endif

      return
      end function INTERPOL
      !***

      subroutine WRITE_OUTPUT( npx , npy , buf                    &
                             , npv                                &
                             , gx , gy                            &
                             , path                               &
                             , var, e, p )

      implicit none

      integer              :: npx, npy
      integer              :: buf
      integer              :: npv

      real                 :: gx (1:npx+2*buf)
      real                 :: gy (1:npy+2*buf)
      real, optional       :: var(1:npx+2*buf,1:npy+2*buf,npv), e(1:npx+2*buf,1:npy+2*buf), p(1:npx+2*buf,1:npy+2*buf)

      character(len=*)   :: path

      integer              :: i , j

      integer              :: iunit = 100
      character(len=400)   :: datafile

      logical              :: centered = .true.

      datafile = trim(path)//'/boundary_input_test.dat'

      open( iunit , file=datafile )


      write(iunit ,'(a)') 'VARIABLES =  "y" , "z" , "u" , "v" , "w" , "rho" , "T" ,'   &
                                        //' "R11" , "R22" , "R33" , "R21" , "R31" , "R32", "rhoEt", "p" '
      write(iunit ,'(a,I0,a,I0,a,I0,a)') 'Zone T = B0001 I = ', npx+2*buf              &
                                         ,          ' ,  J = ', npy+2*buf,' ,  F = BLOCK '

      write(iunit ,'(6(4X,E18.8))') (( gx(i) , i=1,npx+2*buf ), j=1,npy+2*buf )
      write(iunit ,'(6(4X,E18.8))') (( gy(j) , i=1,npx+2*buf ), j=1,npy+2*buf )


      do l = 1 , npv
        write(iunit,'(6(4X,E18.8))') (( var(i,j,l) , i=1,npx+2*buf ), j=1,npy+2*buf )
      end do
      write(iunit,'(6(4X,E18.8))') (( e(i,j) , i=1,npx+2*buf ), j=1,npy+2*buf )
      write(iunit,'(6(4X,E18.8))') (( p(i,j) , i=1,npx+2*buf ), j=1,npy+2*buf )

      close (iunit)

      end subroutine WRITE_OUTPUT



      ! print an abort message in case anything goes wrong
      subroutine ABORT( message )
          implicit none

          character ( len = * ),optional :: message

          print *, 'FATAL'
          print *, 'FATAL a critical error occured'
          print *, 'FATAL'

          if ( present(message) ) then
              print *, 'FATAL ',TRIM( message )
              print *, 'FATAL'
              print *, ''
          end if

          stop "It's time to pull the emergency brake."

      end subroutine ABORT

      end program inflow_field

! cttblx_dat*.dat
!(1  )        "y"
!(2  )        "y_plus"
!(3  )        "u_plus_v"
!(4  )        "u_plus_log"
!(5  )        "u_plus"
!(6  )        "uVD/utau"
!(7  )        "M_loc"
!(8  )        "Mt"
!(9  )        "Tt_fav"
!(10 )        "Tt"
!(11 )        "Prt_fav"
!(12 )        "Prt"
!(13 )        "sra_fav"
!(14 )        "hsra_fav"
!(15 )        "hsra"
!(16 )        "T_fav"
!(17 )        "-R_uT"
!(18 )        "utau"
!(19 )        "Tt_fluc_o"
!(20 )        "Tt_fluc_fav"
!(21 )        "Tt_fluc"
!(22 )        "<u>"
!(23 )        "<v>"
!(24 )        "<w>"
!(25 )        "<rho>"
!(26 )        "<energy>"
!(27 )        "<c>"
!(28 )        "<p>"
!(29 )        "<T>"
!(30 )        "<visc>"
!(31 )        "<uu>"
!(32 )        "<vv>"
!(33 )        "<ww>"
!(34 )        "<uv>"
!(35 )        "<uu>/ut2"
!(36 )        "<vv>/ut2"
!(37 )        "<ww>/ut2"
!(38 )        "<uv>/ut2"
!(39 )        "<uu>*r/rw*1./ut2"
!(40 )        "<vv>*r/rw*1./ut2"
!(41 )        "<ww>*r/rw*1./ut2"
!(42 )        "<uv>*r/rw*1./ut2"
!(43 )        "<uT>"
!(44 )        "<vT>"
!(45 )        "<rr>"
!(46 )        "<TT>"
!(47 )        "<EE>"
!(48 )        "<r`r`/rho>"
!(49 )        "<pp>"



