# cmake/ConfigureSummary.cmake
#
# This file is part of NEST GPU.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST GPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST GPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.

function( NEST_PRINT_CONFIG_SUMMARY )
  # set summary color: here please choose appropriate color!
  message( "${BoldCyan}" )
  message( "--------------------------------------------------------------------------------" )
  message( "NEST GPU Configuration Summary" )
  message( "--------------------------------------------------------------------------------" )
  message( "" )
  if ( CMAKE_BUILD_TYPE )
    message( "Build type          : ${CMAKE_BUILD_TYPE}" )
  endif ()
  message( "Target System       : ${CMAKE_SYSTEM_NAME}" )
  message( "Cross Compiling     : ${CMAKE_CROSSCOMPILING}" )
  message( "C compiler          : ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION} (${CMAKE_C_COMPILER})" )
  message( "C compiler flags    : ${ALL_CFLAGS}" )
  message( "C++ compiler        : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CXX_COMPILER})" )
  message( "C++ compiler flags  : ${ALL_CXXFLAGS}" )
  message( "CUDA compiler        : ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION} (${CMAKE_CUDA_COMPILER})" )
  message( "CUDA compiler flags  : ${ALL_CUDA_FLAGS}" )
  message( "Build dynamic       : ${BUILD_SHARED_LIBS}" )

  message( "" )
  message( "Python bindings     : Yes (Python ${Python_VERSION}: ${PYTHON})" )
  message( "    Includes        : ${Python_INCLUDE_DIRS}" )
  message( "    Libraries       : ${Python_LIBRARIES}" )

  message( "" )
  if ( HAVE_LIBLTDL )
    message( "Use libltdl         : Yes (LTDL ${LTDL_VERSION})" )
    message( "    Includes        : ${LTDL_INCLUDE_DIRS}" )
    message( "    Libraries       : ${LTDL_LIBRARIES}" )
  else ()
    message( "Use libltdl         : No" )
  endif ()

  #message( "" )
  #if ( DOXYGEN_FOUND )
  #  message( "Use doxygen         : Yes (${DOXYGEN_EXECUTABLE}); make target `doc` is available" )
  #  if ( DOXYGEN_DOT_FOUND )
  #    message( "    Graphviz        : Yes (${DOXYGEN_DOT_EXECUTABLE}); make target `fulldoc` is available" )
  #  endif ()
  #else ()
  #  message( "Use doxygen         : No" )
  #endif ()

  message( "" )
  if ( HAVE_MPI )
    message( "Use MPI             : Yes (MPI: ${MPI_CXX_COMPILER})" )
    message( "    Includes        : ${MPI_CXX_INCLUDE_PATH}" )
    message( "    Libraries       : ${MPI_CXX_LIBRARIES}" )
    if ( MPI_CXX_COMPILE_FLAGS )
      message( "    Compile Flags   : ${MPI_CXX_COMPILE_FLAGS}" )
    endif ()
    if ( MPI_CXX_LINK_FLAGS )
      message( "    Link Flags      : ${MPI_CXX_LINK_FLAGS}" )
    endif ()
    set( MPI_LAUNCHER "${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} <np> ${MPIEXEC_PREFLAGS} <prog> ${MPIEXEC_POSTFLAGS} <args>" )
    string(REPLACE "  " " " MPI_LAUNCHER ${MPI_LAUNCHER})
    message( "    Launcher        : ${MPI_LAUNCHER}")
  else ()
    message( "Use MPI             : No" )
  endif ()

  if ( with-libraries )
    message( "" )
    message( "Additional libraries:" )
    foreach ( lib ${with-libraries} )
      message( "                     ${lib}" )
    endforeach ()
  endif ()

  if ( with-includes )
    message( "" )
    message( "Additional includes:" )
    foreach ( inc ${with-includes} )
      message( "                     -I${inc}" )
    endforeach ()
  endif ()

  if ( HAVE_MPI )
    message( "" )
    message( "For details on setting specific flags for your MPI launcher command, see the" )
    message( "CMake documentation at https://cmake.org/cmake/help/latest/module/FindMPI.html")
  endif ()

  message( "" )
  message( "--------------------------------------------------------------------------------" )
  message( "" )
  message( "The NEST GPU executable will be installed to:" )
  message( "  ${CMAKE_INSTALL_FULL_BINDIR}/" )
  message( "" )
  message( "NEST dynamic libraries and user modules will be installed to:" )
  message( "  ${CMAKE_INSTALL_FULL_LIBDIR}/nest/" )
  message( "" )
  message( "Documentation and examples will be installed to:" )
  message( "  ${CMAKE_INSTALL_FULL_DOCDIR}/" )
  message( "" )
  message( "PyNEST will be installed to:" )
  message( "    ${CMAKE_INSTALL_PREFIX}/${PYEXECDIR}" )
  message( "" )
  message( "To set necessary environment variables, add the following line" )
  message( "to your ~/.bashrc :" )
  message( "  source ${CMAKE_INSTALL_FULL_BINDIR}/nestgpu_vars.sh" )
  message( "" )
  message( "--------------------------------------------------------------------------------" )
  message( "" )

  message( "You can now build and install NEST GPU with" )
  message( "  make" )
  message( "  make install" )
  message( "  make installcheck" )
  message( "" )

  message( "If you experience problems with the installation or the use of NEST GPU," )
  message( "go to https://nest-gpu.readthedocs.io/en/latest/community.html to find out how to" )
  message( "join the user mailing list." )

  # reset output color
  message( "${Reset}" )
endfunction()
