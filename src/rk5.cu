/*
 *  rk5.cu
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
 * @file rk5.cu
 * @brief Runge-Kutta-Fehlberg 5th order numerical integration for NEST GPU
 *
 * This file implements the RK5 (Runge-Kutta-Fehlberg 5th order) adaptive
 * stepsize numerical integration method, which provides high accuracy for
 * solving stiff differential equations in neuron models.
 *
 * Mathematical Foundation:
 * -----------------------
 * The RK5 method is a 6th-order Runge-Kutta method with 5th-order error
 * estimation, providing adaptive stepsize control for optimal accuracy
 * and efficiency.
 *
 * Butcher tableau coefficients:
 * - c1-c6: Time step coefficients
 * - a21-a65: Runge-Kutta matrix coefficients
 * - b1-b7: Solution combination coefficients
 * - dc1-dc7: Error estimation coefficients
 *
 * Key Features:
 * -------------
 * - Adaptive stepsize control
 * - High accuracy (5th order)
 * - Error estimation for step adjustment
 * - Suitable for stiff equations
 * - Numerical stability
 *
 * GPU Implementation:
 * ------------------
 * CUDA implementation for parallel integration:
 * - Multiple neurons integrated simultaneously
 * - Efficient memory access patterns
 * - Minimal thread divergence
 * - Coalesced global memory access
 *
 * Integration Method:
 * ------------------
 * The algorithm:
 * 1. Calculate six intermediate stages
 * 2. Combine stages for 5th-order solution
 * 3. Calculate error estimate
 * 4. Adjust stepsize based on error tolerance
 * 5. Repeat if error too large
 *
 * Error Control:
 * -------------
 * - Local truncation error monitored
 * - Stepsize adjusted adaptively
 * - Maintains specified error tolerance
 * - Balances accuracy and performance
 *
 * Applications:
 * -------------
 * - Adaptive exponential neuron models
 * - Conductance-based synapses
 * - Stiff differential equations
 * - High-precision requirements
 *
 * Performance:
 * ------------
 * - Higher accuracy than Euler method
 * - More computation per step
 * - Fewer steps due to adaptive control
 * - Overall efficient for stiff systems
 *
 * Integration Points:
 * ------------------
 * - Neuron models: aeif_cond_beta, aeif_psc_exp
 * - Differential equation solvers
 * - Adaptive stepsize control systems
 *
 * Scientific Background:
 * ----------------------
 * Based on Fehlberg (1969):
 * "Classical Fifth-, Sixth-, Seventh-, and Eighth-Order
 * Runge-Kutta Formulas with Stepsize Control"
 *
 * The RK5 method is widely used in computational neuroscience
 * for its balance of accuracy and efficiency with stiff
 * neuronal dynamics.
 *
 * @see rk5.h RK5 interface and declarations
 * @see neuron_models.h Models using RK5 integration
 */

#include "rk5.h"
#include <cmath>
#include <config.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdio.h>

__constant__ float c2 = 0.2;
__constant__ float c3 = 0.3;
__constant__ float c4 = 0.6;
__constant__ float c5 = 1.0;
__constant__ float c6 = 0.875;
__constant__ float a21 = 0.2;
__constant__ float a31 = 3.0 / 40.0;
__constant__ float a32 = 9.0 / 40.0;
__constant__ float a41 = 0.3;
__constant__ float a42 = -0.9;
__constant__ float a43 = 1.2;
__constant__ float a51 = -11.0 / 54.0;
__constant__ float a52 = 2.5;
__constant__ float a53 = -70.0 / 27.0;
__constant__ float a54 = 35.0 / 27.0;
__constant__ float a61 = 1631.0 / 55296.0;
__constant__ float a62 = 175.0 / 512.0;
__constant__ float a63 = 575.0 / 13824.0;
__constant__ float a64 = 44275.0 / 110592.0;
__constant__ float a65 = 253.0 / 4096.0;

__constant__ float a71 = 37.0 / 378.0;
__constant__ float a73 = 250.0 / 621.0;
__constant__ float a74 = 125.0 / 594.0;
__constant__ float a76 = 512.0 / 1771.0;

__constant__ float e1 = 37.0 / 378.0 - 2825.0 / 27648.0;
__constant__ float e3 = 250.0 / 621.0 - 18575.0 / 48384.0;
__constant__ float e4 = 125.0 / 594.0 - 13525.0 / 55296.0;
__constant__ float e5 = -277.00 / 14336.0;
__constant__ float e6 = 512.0 / 1771.0 - 0.25;

__constant__ float eps = 1.0e-6;
__constant__ float coeff = 0.9;
__constant__ float exp_inc = -0.2;
__constant__ float exp_dec = -0.25;
__constant__ float err_min = 1.889568e-4; //(5/coeff)^(1/exp_inc)
__constant__ float scal_min = 1.0e-1;

__global__ void
SetFloatArray( float* arr, int n_elem, int step, float val )
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if ( array_idx < n_elem )
  {
    arr[ array_idx * step ] = val;
  }
}
