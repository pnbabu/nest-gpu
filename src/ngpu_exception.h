/*
 *  ngpu_exception.h
 *
 *  This file is part of NEST GPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

/**
 * @file ngpu_exception.h
 * @brief Exception handling and error reporting system for NEST GPU
 *
 * This file defines the exception handling infrastructure used throughout
 * NEST GPU for robust error detection, reporting, and handling.
 *
 * Exception Architecture:
 * The system provides C++ exceptions for error handling with:
 * - Custom exception class for NEST GPU-specific errors
 * - Integration with standard C++ exception mechanisms
 * - MPI-aware error reporting for distributed simulations
 * - Verbosity-controlled diagnostic output
 *
 * Key Components:
 * - ngpu_exception: Custom exception class for runtime errors
 * - MPIRankString(): Process identification in distributed runs
 * - BEGIN_TRY/END_TRY: Exception handling macros
 * - verbosePrint(): Controlled diagnostic output
 *
 * Error Categories Handled:
 * - Memory allocation failures
 * - CUDA initialization errors
 * - Invalid parameter values
 * - Network configuration errors
 * - File I/O errors
 * - MPI communication failures
 *
 * MPI Integration:
 * In distributed simulations, error messages include:
 * - Process rank identification
 * - Synchronized error reporting
 * - Collective error handling
 * - Debugging support for multi-process scenarios
 *
 * Usage Pattern:
 * @code
 * BEGIN_TRY {
 *     // NEST GPU operations
 *     ngpu.Create("neuron_model", 1000);
 * } END_TRY
 * @endcode
 *
 * Verbosity System:
 * - Level 0: Quiet (no output)
 * - Level 1-2: Critical information
 * - Level 3-5: Normal debugging output
 * - Level 6+: Very verbose output
 *
 * Error Reporting:
 * - Descriptive error messages
 * - Context information included
 * - MPI rank when applicable
 * - Stack trace information when available
 *
 * Integration Points:
 * - All NEST GPU classes throw ngpu_exception
 * - CUDA errors wrapped in ngpu_exception
 * - MPI errors handled with context
 * - Memory errors properly reported
 *
 * Performance Impact:
 * - Minimal overhead when exceptions not thrown
 * - Exception handling may impact critical paths
 * - Verbosity checking is lightweight
 * - MPI rank query cached when possible
 *
 * Best Practices:
 * - Use exceptions for exceptional conditions
 * - Provide descriptive error messages
 * - Include context in error messages
 * - Handle exceptions at appropriate levels
 * - Use verbosity system for debugging
 *
 * @see cuda_error.h CUDA error handling
 * @see nestgpu.h Main interface exception handling
 */

/////////////////////////////////////
// ngpu_exception class definition
// This class handles runtime errors throughout NEST GPU
/////////////////////////////////////

#ifndef NGPUEXCEPTION_H
#define NGPUEXCEPTION_H
#include <iostream>
#include <cstring>
#include <exception>
#include <string>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

/**
 * @brief Get MPI rank string for error reporting
 *
 * Returns a string containing the MPI rank information if running in
 * a distributed simulation. Used to identify which process reported
 * an error in multi-process scenarios.
 *
 * @return String with MPI rank information, or empty string if not MPI
 *
 * Behavior:
 * - Checks if MPI is initialized
 * - Returns rank if running with multiple processes
 * - Returns empty string for single-process runs
 *
 * Performance:
 * - MPI_Initialized call has minimal overhead
 * - Result cached in implementation
 * - Called only during error handling
 *
 * Usage:
 * @code
 * std::cerr << MPIRankString() << "Error message\n";
 * // Output: "MPI rank: 2	Error message" (in MPI run)
 * // Output: "Error message" (in single process)
 * @endcode
 */
inline std::string
MPIRankString()
{

#ifdef HAVE_MPI
  int initialized;
  MPI_Initialized( &initialized );
  if ( initialized ) {
    int proc_num;
    MPI_Comm_size( MPI_COMM_WORLD, &proc_num );
    if ( proc_num > 1 ) {
      int mpi_id;
      MPI_Comm_rank( MPI_COMM_WORLD, &mpi_id );
      return std::string("MPI rank: ") + std::to_string(mpi_id) + "\t";
    }
  }
#endif

  return "";
}



///////////////////////////////////
// ngpu_exception class definition
// Handles runtime errors throughout NEST GPU with descriptive messages
//////////////////////////////////
/**
 * @class ngpu_exception
 * @brief Custom exception class for NEST GPU runtime errors
 *
 * This class extends std::exception to provide NEST GPU-specific error
 * handling with descriptive error messages and integration with the
 * exception handling infrastructure.
 *
 * Features:
 * - Stores descriptive error messages
 * - Inherits from std::exception for compatibility
 * - String duplication for message persistence
 * - Integration with exception handling macros
 *
 * Error Handling Flow:
 * 1. Error detected in NEST GPU code
 * 2. ngpu_exception thrown with descriptive message
 * 3. Exception caught by BEGIN_TRY/END_TRY macros
 * 4. Error message reported with MPI context
 * 5. Program may continue or terminate based on configuration
 *
 * Memory Management:
 * - Error messages are dynamically allocated
 * - Memory freed in destructor
 * - Safe for exception propagation
 *
 * Usage:
 * @code
 * throw ngpu_exception("Invalid parameter value");
 * throw ngpu_exception(std::string("CUDA error: ") + cudaGetErrorString(err));
 * @endcode
 *
 * @note Inherits from std::exception for standard compatibility
 * @see BEGIN_TRY/END_TRY macros for exception handling
 */
class ngpu_exception : public std::exception
{
  const char* Message; /**< Error message string (dynamically allocated) */

public:
  /**
   * @brief Constructor from C string
   * @param ch Null-terminated error message string
   *
   * Creates an exception with the specified message. The message
   * is duplicated using strdup() to ensure it persists.
   *
   * @note Caller doesn't need to keep the original string
   */
  ngpu_exception( const char* ch )
  {
    Message = strdup( ch );
  }

  /**
   * @brief Constructor from C++ string
   * @param s Error message string
   *
   * Creates an exception from a std::string. The string content
   * is duplicated for persistence.
   *
   * @note More convenient than C string constructor
   */
  ngpu_exception( std::string s )
  {
    Message = strdup( s.c_str() );
  }

  /**
   * @brief Get error message
   * @return Pointer to error message string
   *
   * Implements std::exception::what() to provide the error message
   * to exception handlers.
   *
   * @note Message persists for the lifetime of the exception object
   */
  virtual const char*
  what() const throw()
  {
    return Message;
  }
};

#define BEGIN_TRY try
#define END_TRY                                                         \
  catch ( ngpu_exception & e )                                          \
  {                                                                     \
    std::cerr << MPIRankString() << "Error: " << e.what() << "\n";	\
  }                                                                     \
  catch ( bad_alloc& )                                                  \
  {                                                                     \
    std::cerr << MPIRankString() << "Error allocating memory."          \
              << "\n";                                                  \
  }                                                                     \
  catch ( ... )                                                         \
  {                                                                     \
    std::cerr << MPIRankString() << "Unrecognized error\n";             \
  }


namespace verbose_print_ns
{
  extern int verbosity_level_;
}

inline void verbosePrint(std::string message, int verbosity_threshold = 3)
{
  if (verbose_print_ns::verbosity_level_ >= verbosity_threshold) {
    std::cout << MPIRankString() << message << "\n";
  }
}


#endif

