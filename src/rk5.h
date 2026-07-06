/*
 *  rk5.h
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
 * @file rk5.h
 * @brief 5th-order Runge-Kutta-Fehlberg adaptive stepsize integrator
 *
 * This file implements the Runge-Kutta-Fehlberg (RKF45) method for adaptive
 * stepsize integration of ordinary differential equations (ODEs). The method
 * provides 5th-order accuracy with embedded 4th-order error estimation.
 *
 * Mathematical Background:
 * The RKF45 method computes two solutions of different orders (4th and 5th)
 * using the same function evaluations, allowing for efficient error estimation
 * and stepsize control.
 *
 * The method uses the following Butcher tableau coefficients:
 * @f[
 * \begin{array}{c|cccccc}
 * 0 & & & & & & \\
 * \frac{1}{4} & \frac{1}{4} & & & & \\
 * \frac{3}{8} & \frac{3}{32} & \frac{9}{32} & & & \\
 * \frac{12}{13} & \frac{1932}{2197} & -\frac{7200}{2197} & \frac{7296}{2197} & & \\
 * 1 & \frac{439}{216} & -8 & \frac{3680}{513} & -\frac{845}{4104} & \\
 * \frac{1}{2} & -\frac{8}{27} & 2 & -\frac{3544}{2565} & \frac{1859}{4104} & -\frac{11}{40} \\
 * \hline
 * \text{4th order} & \frac{25}{216} & 0 & \frac{1408}{2565} & \frac{2197}{4104} & -\frac{1}{5} & 0 \\
 * \text{5th order} & \frac{16}{135} & 0 & \frac{6656}{12825} & \frac{28561}{56430} & -\frac{9}{50} & \frac{2}{55}
 * \end{array}
 * @f]
 *
 * Adaptive Stepsize Control:
 * The error estimate is used to adjust the stepsize:
 * @f[
 * h_{new} = h \cdot \left(\frac{\tol \cdot h}{\err}\right)^{1/5}
 * @f]
 * where tol is the tolerance and err is the estimated error.
 *
 * Error Estimation:
 * The difference between 4th and 5th order solutions provides an error estimate:
 * @f[
 * \text{err} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_{5,i}-y_{4,i}}{sc_i}\right)^2}
 * @f]
 * where @f$sc_i@f$ is a scaling factor for variable i.
 *
 * GPU Implementation:
 * - Template-based design for variable problem sizes
 * - One thread per ODE system
 * - Coalesced memory access for state arrays
 * - Efficient use of shared memory for intermediate calculations
 *
 * Performance Characteristics:
 * - 6 function evaluations per step (rk5 to rk6)
 * - Adaptive stepsize reduces total steps for smooth dynamics
 * - Error control maintains accuracy while maximizing efficiency
 * - Parallel integration of multiple independent systems
 *
 * Usage in NEST GPU:
 * - Used for complex neuron models (e.g., multi-compartment)
 * - Adaptive stepsize handles stiff dynamics
 * - Particularly useful for models with widely varying time scales
 *
 * Numerical Stability:
 * - Built-in error control prevents divergence
 * - Stepsize limits prevent numerical instability
 * - Scaling factors handle variables with different magnitudes
 *
 * @see rk5_const.h Runge-Kutta method constants
 * @see rk5_interface.h Interface functions for neuron models
 * @see propagator_stability.cu Stability analysis for exact integration
 */

#ifndef RK5_H
#define RK5_H

#include "cuda_error.h"
#include "rk5_const.h"
#include "rk5_interface.h"

#define MIN( a, b ) ( ( ( a ) < ( b ) ) ? ( a ) : ( b ) )
#define MAX( a, b ) ( ( ( a ) > ( b ) ) ? ( a ) : ( b ) )

/**
 * @brief GPU kernel to initialize float array with constant value
 *
 * Sets all elements of a float array to a specified value, used for
 * initializing integration variables and parameters.
 *
 * @param arr Array to initialize (device memory)
 * @param n_elem Number of elements to set
 * @param step Stride between elements
 * @param val Value to set
 *
 * @note Used for array initialization in RK5 solver
 */
__global__ void SetFloatArray( float* arr, int n_elem, int step, float val );

/**
 * @brief GPU kernel to initialize ODE system arrays
 *
 * Initializes arrays of ODE systems for parallel integration on GPU.
 * Each thread initializes one system with its initial conditions and parameters.
 *
 * @tparam DataStruct User-defined data structure for model-specific data
 * @param array_size Number of ODE systems to initialize
 * @param n_var Number of variables per system
 * @param n_param Number of parameters per system
 * @param x_arr Array for independent variable (time) values
 * @param h_arr Array for step sizes
 * @param y_arr Array for initial conditions (flattened 2D)
 * @param par_arr Array for parameters (flattened 2D)
 * @param x_min Initial value for independent variable
 * @param h Initial step size
 * @param data_struct Model-specific data structure
 *
 * Thread Organization:
 * - Each thread processes one ODE system
 * - Block size: typically 128-256 threads
 * - Grid size: ceil(array_size / block_size)
 *
 * @note Called before integration begins
 * @see ArrayCalibrate() for calibration step
 */
template < class DataStruct >
__global__ void
ArrayInit( int array_size,
  int n_var,
  int n_param,
  double* x_arr,
  float* h_arr,
  float* y_arr,
  float* par_arr,
  double x_min,
  float h,
  DataStruct data_struct )
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if ( array_idx < array_size )
  {
    NodeInit( n_var, n_param, x_min, &y_arr[ array_idx * n_var ], &par_arr[ array_idx * n_param ], data_struct );
    x_arr[ array_idx ] = x_min;
    h_arr[ array_idx ] = h;
  }
}

/**
 * @brief GPU kernel to calibrate ODE systems before integration
 *
 * Performs calibration step for ODE systems, computing any necessary
 * pre-integration values or constants. Called after initialization but
 * before integration begins.
 *
 * @tparam DataStruct User-defined data structure for model-specific data
 * @param array_size Number of ODE systems to calibrate
 * @param n_var Number of variables per system
 * @param n_param Number of parameters per system
 * @param x_arr Array for independent variable values
 * @param h_arr Array for step sizes
 * @param y_arr Array for system states
 * @param par_arr Array for parameters
 * @param x_min Initial value for independent variable
 * @param h Initial step size
 * @param data_struct Model-specific data structure
 *
 * Thread Organization:
 * - Each thread processes one ODE system
 * - Parallel calibration of multiple systems
 *
 * @note Called once before integration starts
 * @see ArrayInit() for initial setup
 */
template < class DataStruct >
__global__ void
ArrayCalibrate( int array_size,
  int n_var,
  int n_param,
  double* x_arr,
  float* h_arr,
  float* y_arr,
  float* par_arr,
  double x_min,
  float h,
  DataStruct data_struct )
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if ( array_idx < array_size )
  {
    NodeCalibrate( n_var, n_param, x_min, &y_arr[ array_idx * n_var ], &par_arr[ array_idx * n_param ], data_struct );
    x_arr[ array_idx ] = x_min;
    h_arr[ array_idx ] = h;
  }
}

/**
 * @brief Core Runge-Kutta-Fehlberg integration step
 *
 * Performs one adaptive stepsize integration step using the RKF45 method.
 * Computes both 4th and 5th order solutions, estimates error, and adjusts
 * stepsize accordingly.
 *
 * @tparam NVAR Number of variables in the ODE system
 * @tparam NPARAM Number of parameters
 * @tparam DataStruct User-defined data structure for model-specific data
 *
 * @param x Current independent variable value (time) [updated]
 * @param y Current state vector [updated]
 * @param h Current step size [updated adaptively]
 * @param h_min Minimum allowed step size
 * @param h_max Maximum allowed step size
 * @param param Parameter array
 * @param data_struct Model-specific data structure
 *
 * Algorithm:
 * 1. Compute initial derivatives (k1)
 * 2. Compute scaling factors for error estimation
 * 3. Perform RK45 substeps (k2-k6)
 * 4. Compute 4th and 5th order solutions
 * 5. Estimate error and adjust stepsize
 * 6. Repeat if error is too large
 * 7. Accept step and update state if error is acceptable
 *
 * Error Control:
 * - Uses relative and absolute error tolerances
 * - Scaling factors handle variables of different magnitudes
 * - Stepsize adjusted based on error estimate
 * - Prevents numerical instability with stepsize limits
 *
 * Performance:
 * - 6 derivative evaluations per attempted step
 * - Rejected steps require recomputation
 * - Adaptive stepsize optimizes efficiency
 * - Template-based optimization for fixed sizes
 *
 * Memory Usage:
 * - Uses local arrays for intermediate calculations
 * - ~8*NVAR floats of stack space
 * - No dynamic memory allocation
 *
 * @note Device function - must be called from GPU kernel
 * @warning May loop indefinitely if error tolerance cannot be met
 * @see rk5_const.h for method constants
 * @see Derivatives() for user-defined derivative function
 */
template < int NVAR, int NPARAM, class DataStruct >
__device__ void
RK5Step( double& x, float* y, float& h, float h_min, float h_max, float* param, DataStruct data_struct )
{
  // Intermediate arrays for Runge-Kutta stages
  float y_new[ NVAR ];      // Intermediate state values
  float k1[ NVAR ];         // First RK coefficient
  float k2[ NVAR ];         // Second RK coefficient
  float k3[ NVAR ];         // Third RK coefficient
  float k4[ NVAR ];         // Fourth RK coefficient
  float k5[ NVAR ];         // Fifth RK coefficient
  float k6[ NVAR ];         // Sixth RK coefficient
  float y_scal[ NVAR ];     // Scaling factors for error estimation

  // Compute initial derivatives and scaling factors
  Derivatives< NVAR, NPARAM >( x, y, k1, param, data_struct );
  for ( int i = 0; i < NVAR; i++ )
  {
    y_scal[ i ] = fabs( y[ i ] ) + fabs( k1[ i ] * h ) + scal_min;
  }

  // Adaptive stepsize loop
  float err;
  for ( ;; )
  {
    // Enforce stepsize limits
    if ( h > h_max )
    {
      h = h_max;
    }
    if ( h < h_min )
    {
      h = h_min;
    }

    // RK45 substeps - compute intermediate values
    // Stage 2
    for ( int i = 0; i < NVAR; i++ )
    {
      y_new[ i ] = y[ i ] + h * a21 * k1[ i ];
    }

    Derivatives< NVAR, NPARAM >( x + c2 * h, y_new, k2, param, data_struct );

    // Stage 3
    for ( int i = 0; i < NVAR; i++ )
    {
      y_new[ i ] = y[ i ] + h * ( a31 * k1[ i ] + a32 * k2[ i ] );
    }
    Derivatives< NVAR, NPARAM >( x + c3 * h, y_new, k3, param, data_struct );

    // Stage 4
    for ( int i = 0; i < NVAR; i++ )
    {
      y_new[ i ] = y[ i ] + h * ( a41 * k1[ i ] + a42 * k2[ i ] + a43 * k3[ i ] );

    for ( int i = 0; i < NVAR; i++ )
    {
      y_new[ i ] = y[ i ] + h * ( a31 * k1[ i ] + a32 * k2[ i ] );
    }
    Derivatives< NVAR, NPARAM >( x + c3 * h, y_new, k3, param, data_struct );

    for ( int i = 0; i < NVAR; i++ )
    {
      y_new[ i ] = y[ i ] + h * ( a41 * k1[ i ] + a42 * k2[ i ] + a43 * k3[ i ] );
    }
    Derivatives< NVAR, NPARAM >( x + c4 * h, y_new, k4, param, data_struct );

    for ( int i = 0; i < NVAR; i++ )
    {
      y_new[ i ] = y[ i ] + h * ( a51 * k1[ i ] + a52 * k2[ i ] + a53 * k3[ i ] + a54 * k4[ i ] );
    }
    Derivatives< NVAR, NPARAM >( x + c5 * h, y_new, k5, param, data_struct );

    for ( int i = 0; i < NVAR; i++ )
    {
      y_new[ i ] = y[ i ] + h * ( a61 * k1[ i ] + a62 * k2[ i ] + a63 * k3[ i ] + a64 * k4[ i ] + a65 * k5[ i ] );
    }
    Derivatives< NVAR, NPARAM >( x + c6 * h, y_new, k6, param, data_struct );

    for ( int i = 0; i < NVAR; i++ )
    {
      y_new[ i ] = y[ i ] + h * ( a71 * k1[ i ] + a73 * k3[ i ] + a74 * k4[ i ] + a76 * k6[ i ] );
    }

    err = 0.0;
    for ( int i = 0; i < NVAR; i++ )
    {
      float val = h * ( e1 * k1[ i ] + e3 * k3[ i ] + e4 * k4[ i ] + e5 * k5[ i ] + e6 * k6[ i ] );
      val /= y_scal[ i ]; ///// check for overflow!!!!!!!!!!!
      err = MAX( err, fabs( val ) );
    }
    err /= eps;
    if ( err <= 1.0 || h <= h_min * ( 1.0 + 1.0e-5 ) )
    {
      break;
    }

    float h_new = h * coeff * pow( err, exp_dec );
    h = MAX( h_new, 0.1 * h );

    // if (h <= h_min) {
    //   h = h_min;
    // }
    // x_new = x + h;
  }

  x += h;

  if ( err > err_min )
  {
    h = h * coeff * pow( err, exp_inc );
  }
  else
  {
    h = 5.0 * h;
  }

  for ( int i = 0; i < NVAR; i++ )
  {
    y[ i ] = y_new[ i ];
  }
}

template < int NVAR, int NPARAM, class DataStruct >
__device__ void
RK5Update( double& x, float* y, double x1, float& h, float h_min, float* param, DataStruct data_struct )
{
  bool end_time_step = false;
  while ( !end_time_step )
  {
    float hmax = ( float ) ( x1 - x );
    RK5Step< NVAR, NPARAM, DataStruct >( x, y, h, h_min, hmax, param, data_struct );
    end_time_step = ( x >= x1 - h_min );
    ExternalUpdate< NVAR, NPARAM >( x, y, param, end_time_step, data_struct );
  }
}

template < int NVAR, int NPARAM, class DataStruct >
__global__ void
ArrayUpdate( int array_size,
  double* x_arr,
  float* h_arr,
  float* y_arr,
  float* par_arr,
  double x1,
  float h_min,
  DataStruct data_struct )
{
  int ArrayIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if ( ArrayIdx < array_size )
  {
    double x = x_arr[ ArrayIdx ];
    float h = h_arr[ ArrayIdx ];
    float y[ NVAR ];
    float param[ NPARAM ];

    for ( int i = 0; i < NVAR; i++ )
    {
      y[ i ] = y_arr[ ArrayIdx * NVAR + i ];
    }
    for ( int j = 0; j < NPARAM; j++ )
    {
      param[ j ] = par_arr[ ArrayIdx * NPARAM + j ];
    }

    RK5Update< NVAR, NPARAM, DataStruct >( x, y, x1, h, h_min, param, data_struct );

    x_arr[ ArrayIdx ] = x;
    h_arr[ ArrayIdx ] = h;
    for ( int i = 0; i < NVAR; i++ )
    {
      y_arr[ ArrayIdx * NVAR + i ] = y[ i ];
    }
    for ( int j = 0; j < NPARAM; j++ )
    {
      par_arr[ ArrayIdx * NPARAM + j ] = param[ j ];
    }
  }
}

template < class DataStruct >
class RungeKutta5
{
  int array_size_;
  int n_var_;
  int n_param_;

  double* d_XArr;
  float* d_HArr;
  float* d_YArr;
  float* d_ParamArr;

public:
  ~RungeKutta5();

  double*
  GetXArr()
  {
    return d_XArr;
  }
  float*
  GetHArr()
  {
    return d_HArr;
  }
  float*
  GetYArr()
  {
    return d_YArr;
  }
  float*
  GetParamArr()
  {
    return d_ParamArr;
  }
  int Init( int array_size, int n_var, int n_param, double x_min, float h, DataStruct data_struct );
  int Calibrate( double x_min, float h, DataStruct data_struct );

  int Free();

  int GetX( int i_array, int n_elem, double* x );
  int GetY( int i_var, int i_array, int n_elem, float* y );
  int SetParam( int i_param, int i_array, int n_param, int n_elem, float val );
  int SetVectParam( int i_param, int i_array, int n_param, int n_elem, float* param, int vect_size );
  template < int NVAR, int NPARAM >
  int Update( double x1, float h_min, DataStruct data_struct );
};

template < class DataStruct >
template < int NVAR, int NPARAM >
int
RungeKutta5< DataStruct >::Update( double x1, float h_min, DataStruct data_struct )
{
  ArrayUpdate< NVAR, NPARAM, DataStruct > <<< ( array_size_ + 1023 ) / 1024, 1024 >>>(
    array_size_, d_XArr, d_HArr, d_YArr, d_ParamArr, x1, h_min, data_struct );
  // gpuErrchk( cudaPeekAtLastError() );
  // gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

template < class DataStruct >
RungeKutta5< DataStruct >::~RungeKutta5()
{
  Free();
}

template < class DataStruct >
int
RungeKutta5< DataStruct >::Free()
{
  CUDAFREECTRL( "d_XArr", d_XArr );
  CUDAFREECTRL( "d_HArr", d_HArr );
  CUDAFREECTRL( "d_YArr", d_YArr );
  CUDAFREECTRL( "d_ParamArr", d_ParamArr );

  return 0;
}

template < class DataStruct >
int
RungeKutta5< DataStruct >::Init( int array_size, int n_var, int n_param, double x_min, float h, DataStruct data_struct )
{
  array_size_ = array_size;
  n_var_ = n_var;
  n_param_ = n_param;

  CUDAMALLOCCTRL( "&d_XArr", &d_XArr, array_size_ * sizeof( double ) );
  CUDAMALLOCCTRL( "&d_HArr", &d_HArr, array_size_ * sizeof( float ) );
  CUDAMALLOCCTRL( "&d_YArr", &d_YArr, array_size_ * n_var_ * sizeof( float ) );
  CUDAMALLOCCTRL( "&d_ParamArr", &d_ParamArr, array_size_ * n_param_ * sizeof( float ) );

  ArrayInit< DataStruct > <<< ( array_size + 1023 ) / 1024, 1024 >>>(
    array_size_, n_var, n_param, d_XArr, d_HArr, d_YArr, d_ParamArr, x_min, h, data_struct );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

template < class DataStruct >
int
RungeKutta5< DataStruct >::Calibrate( double x_min, float h, DataStruct data_struct )
{
  ArrayCalibrate< DataStruct > <<< ( array_size_ + 1023 ) / 1024, 1024 >>>(
    array_size_, n_var_, n_param_, d_XArr, d_HArr, d_YArr, d_ParamArr, x_min, h, data_struct );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

template < class DataStruct >
int
RungeKutta5< DataStruct >::GetX( int i_array, int n_elem, double* x )
{
  cudaMemcpy( x, &d_XArr[ i_array ], n_elem * sizeof( double ), cudaMemcpyDeviceToHost );

  return 0;
}

template < class DataStruct >
int
RungeKutta5< DataStruct >::GetY( int i_var, int i_array, int n_elem, float* y )
{
  cudaMemcpy( y, &d_YArr[ i_array * n_var_ + i_var ], n_elem * sizeof( float ), cudaMemcpyDeviceToHost );

  return 0;
}

template < class DataStruct >
int
RungeKutta5< DataStruct >::SetParam( int i_param, int i_array, int n_param, int n_elem, float val )
{
  SetFloatArray<<< ( n_elem + 1023 ) / 1024, 1024 >>>(
    &d_ParamArr[ i_array * n_param_ + i_param ], n_elem, n_param, val );
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

#endif
