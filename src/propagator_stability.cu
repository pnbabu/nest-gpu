/*
 *  propagator_stability.cpp
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
 * @file propagator_stability.cpp
 * @brief Numerical stability functions for exact integration of linear ODEs
 *
 * This file implements numerically stable propagator functions for exact
 * integration of linear ordinary differential equations, particularly
 * for synaptic dynamics in neuron models.
 *
 * Mathematical Foundation:
 * -----------------------
 * Exact integration of linear ODEs using propagator matrices:
 * \[ \frac{dx}{dt} = -\frac{x}{\tau} + f(t) \]
 *
 * The propagator method provides exact solutions:
 * \[ x(t+h) = P_{32} x(t) + \text{particular solution} \]
 *
 * Key Functions:
 * -------------
 * - propagator_32: Stable P32 propagator computation
 * - Handles near-singular cases
 * - Numerical stability for parameter limits
 *
 * Numerical Stability:
 * --------------------
 * The implementation handles:
 * - Singular case: tau_syn = tau
 * - Near-singular cases: |tau_syn - tau| < 0.1
 * - Linear approximation for stability
 * - Exponential computation with expm1()
 *
 * Integration Method:
 * ------------------
 * Exact integration benefits:
 * - No numerical error from time stepping
 * - Exact for time-invariant parameters
 * - Large timesteps possible
 * - High efficiency
 *
 * GPU Implementation:
 * ------------------
 * Device-side computation:
 * - Available in CUDA kernels
 * - Used by neuron update functions
 * - Numerically stable on GPU
 *
 * Applications:
 * -------------
 * - IAF_PSC_EXP neuron model
 * - Synaptic dynamics integration
 * - Linear ODE systems
 * - Propagation of synaptic currents
 *
 * Scientific Background:
 * ----------------------
 * Based on exact integration methods:
 * - Rotter & Diesmann (1999): Exact digital simulation
 * - Klinshov et al. (2014): Stability analysis
 * - Sets of analytic propagators
 *
 * @see propagator_stability.h Stability interface
 * @see iaf_psc_exp.cu Application example
 */

#include "propagator_stability.h"

// C++ includes:
#include <cmath>

// Includes from libnestutil:
// #include "numerics.h"

__device__ double
propagator_32( double tau_syn, double tau, double C, double h )
{
  const double P32_linear = 1.0 / ( 2.0 * C * tau * tau ) * h * h * ( tau_syn - tau ) * exp( -h / tau );
  const double P32_singular = h / C * exp( -h / tau );
  const double P32 =
    -tau / ( C * ( 1.0 - tau / tau_syn ) ) * exp( -h / tau_syn ) * expm1( h * ( 1.0 / tau_syn - 1.0 / tau ) );

  const double dev_P32 = fabs( P32 - P32_singular );

  if ( tau == tau_syn || ( fabs( tau - tau_syn ) < 0.1 && dev_P32 > 2.0 * fabs( P32_linear ) ) )
  {
    return P32_singular;
  }
  else
  {
    return P32;
  }
}

__device__ double
propagator_31( double tau_syn, double tau, double C, double h )
{
  const double P31_linear = 1.0 / ( 3.0 * C * tau * tau ) * h * h * h * ( tau_syn - tau ) * exp( -h / tau );
  const double P31 = 1.0 / C
    * ( exp( -h / tau_syn ) * expm1( -h / tau + h / tau_syn ) / ( tau / tau_syn - 1.0 ) * tau
      - h * exp( -h / tau_syn ) )
    / ( -1.0 - -tau / tau_syn ) * tau;
  const double P31_singular = h * h / 2.0 / C * exp( -h / tau );
  const double dev_P31 = fabs( P31 - P31_singular );

  if ( tau == tau_syn or ( fabs( tau - tau_syn ) < 0.1 and dev_P31 > 2.0 * fabs( P31_linear ) ) )
  {
    return P31_singular;
  }
  else
  {
    return P31;
  }
}
