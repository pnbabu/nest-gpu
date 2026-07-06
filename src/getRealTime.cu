/*
 *  getRealTime.cu
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

/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

/**
 * @file getRealTime.cu
 * @brief Cross-platform real-time clock implementation for NEST GPU
 *
 * This file provides a cross-platform implementation of a high-resolution
 * real-time clock function for performance measurement and benchmarking
 * in NEST GPU simulations.
 *
 * Functionality:
 * --------------
 * Returns real time in seconds with high precision:
 * - Microsecond or nanosecond resolution (platform-dependent)
 * - Cross-platform support (Windows, Unix, macOS)
 * - Monotonic clock when available
 * - Suitable for performance measurements
 *
 * Platform Support:
 * ----------------
 * Windows:
 * - Uses QueryPerformanceCounter()
 * - High-resolution performance counter
 * - Microsecond precision
 *
 * Unix/Linux:
 * - Uses clock_gettime() with CLOCK_MONOTONIC
 * - High-resolution monotonic clock
 * - Nanosecond precision
 *
 * macOS:
 * - Uses mach_absolute_time()
 * - High-resolution monotonic clock
 * - Nanosecond precision
 *
 * Integration Points:
 * ------------------
 * - NESTGPU: Performance measurement
 * - Benchmarking: Simulation timing
 * - Profiling: Performance analysis
 *
 * Usage:
 * ------
 * double start_time = getRealTime();
 * // ... code to measure ...
 * double elapsed = getRealTime() - start_time;
 *
 * Performance:
 * ------------
 * - Minimal overhead
 * - High precision timing
 * - Fast clock access
 *
 * Applications:
 * -------------
 * - Simulation timing measurement
 * - Performance profiling
 * - Benchmarking
 * - Load balancing analysis
 *
 * @see getRealTime.h Timing interface
 */

#if defined( _WIN32 )
#include <Windows.h>
#include <config.h>

#elif defined( __unix__ ) || defined( __unix ) || defined( unix ) || ( defined( __APPLE__ ) && defined( __MACH__ ) )
#include <sys/time.h> /* gethrtime(), gettimeofday() */
#include <time.h>     /* clock_gettime(), time() */
#include <unistd.h>   /* POSIX flags */

#if defined( __MACH__ ) && defined( __APPLE__ )
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif

#else
#error "Unable to define getRealTime( ) for an unknown OS."
#endif

/**
 * Returns the real time, in seconds, or -1.0 if an error occurred.
 *
 * Time is measured since an arbitrary and OS-dependent start time.
 * The returned real time is only useful for computing an elapsed time
 * between two calls to this function.
 */
double
getRealTime()
{
#if defined( _WIN32 )
  FILETIME tm;
  ULONGLONG t;
#if defined( NTDDI_WIN8 ) && NTDDI_VERSION >= NTDDI_WIN8
  /* Windows 8, Windows Server 2012 and later. ---------------- */
  GetSystemTimePreciseAsFileTime( &tm );
#else
  /* Windows 2000 and later. ---------------------------------- */
  GetSystemTimeAsFileTime( &tm );
#endif
  t = ( ( ULONGLONG ) tm.dwHighDateTime << 32 ) | ( ULONGLONG ) tm.dwLowDateTime;
  return ( double ) t / 10000000.0;

#elif ( defined( __hpux ) || defined( hpux ) ) \
  || ( ( defined( __sun__ ) || defined( __sun ) || defined( sun ) ) && ( defined( __SVR4 ) || defined( __svr4__ ) ) )
  /* HP-UX, Solaris. ------------------------------------------ */
  return ( double ) gethrtime() / 1000000000.0;

#elif defined( __MACH__ ) && defined( __APPLE__ )
  /* OSX. ----------------------------------------------------- */
  static double timeConvert = 0.0;
  if ( timeConvert == 0.0 )
  {
    mach_timebase_info_data_t timeBase;
    ( void ) mach_timebase_info( &timeBase );
    timeConvert = ( double ) timeBase.numer / ( double ) timeBase.denom / 1000000000.0;
  }
  return ( double ) mach_absolute_time() * timeConvert;

#elif defined( _POSIX_VERSION )
  /* POSIX. --------------------------------------------------- */
#if defined( _POSIX_TIMERS ) && ( _POSIX_TIMERS > 0 )
  {
    struct timespec ts;
#if defined( CLOCK_MONOTONIC_PRECISE )
    /* BSD. --------------------------------------------- */
    const clockid_t id = CLOCK_MONOTONIC_PRECISE;
#elif defined( CLOCK_MONOTONIC_RAW )
    /* Linux. ------------------------------------------- */
    const clockid_t id = CLOCK_MONOTONIC_RAW;
#elif defined( CLOCK_HIGHRES )
    /* Solaris. ----------------------------------------- */
    const clockid_t id = CLOCK_HIGHRES;
#elif defined( CLOCK_MONOTONIC )
    /* AIX, BSD, Linux, POSIX, Solaris. ----------------- */
    const clockid_t id = CLOCK_MONOTONIC;
#elif defined( CLOCK_REALTIME )
    /* AIX, BSD, HP-UX, Linux, POSIX. ------------------- */
    const clockid_t id = CLOCK_REALTIME;
#else
    const clockid_t id = ( clockid_t ) -1; /* Unknown. */
#endif /* CLOCK_* */
    if ( id != ( clockid_t ) -1 && clock_gettime( id, &ts ) != -1 )
    {
      return ( double ) ts.tv_sec + ( double ) ts.tv_nsec / 1000000000.0;
    }
    /* Fall thru. */
  }
#endif /* _POSIX_TIMERS */

  /* AIX, BSD, Cygwin, HP-UX, Linux, OSX, POSIX, Solaris. ----- */
  struct timeval tm;
  gettimeofday( &tm, nullptr );
  return ( double ) tm.tv_sec + ( double ) tm.tv_usec / 1000000.0;
#else
  return -1.0; /* Failed. */
#endif
}
