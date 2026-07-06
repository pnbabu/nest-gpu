/*
 *  iaf_psc_exp.cu
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
 * @file iaf_psc_exp.cu
 * @brief Leaky integrate-and-fire neuron with exponential postsynaptic currents
 *
 * This file implements the iaf_psc_exp neuron model, a classic
 * leaky integrate-and-fire neuron with current-based synaptic inputs
 * that produce exponential postsynaptic current waveforms.
 *
 * Mathematical Model:
 * -------------------
 * The iaf_psc_exp model implements standard LIF dynamics:
 *
 * Membrane potential equation:
 * \[ \tau_m \frac{dV}{dt} = -(V - E_L) + I_{syn} + I_e \]
 *
 * Where total synaptic current:
 * \[ I_{syn} = I_{ex} + I_{in} \]
 *
 * Exponential PSC dynamics:
 * \[ \tau_{ex} \frac{dI_{ex}}{dt} = -I_{ex} \]
 * \[ \tau_{in} \frac{dI_{in}}{dt} = -I_{in} \]
 *
 * Spike condition and reset:
 * \[ \text{if } V \geq \Theta: V \leftarrow V_{reset}, \quad t_{ref} \text{ refractory period} \]
 *
 * Key Features:
 * -------------
 * - Leaky membrane potential dynamics
 * - Exponential postsynaptic currents
 * - Separate excitatory/inhibitory receptors
 * - Absolute refractory period
 * - Current-based synaptic inputs
 *
 * GPU Implementation:
 * ------------------
 * CUDA kernel for parallel updates:
 * - Each thread updates one neuron
 * - Efficient propagator-based integration
 * - Coalesced memory access patterns
 * - Optimized for large populations
 *
 * Integration Method:
 * ------------------
 * Uses exact integration with propagators:
 * - Analytical solution for linear dynamics
 * - High numerical stability
 * - Exact for time-invariant parameters
 * - Efficient for large populations
 *
 * Model Parameters:
 * ----------------
 * - tau_m: Membrane time constant (ms)
 * - C_m: Membrane capacitance (pF)
 * - E_L: Resting potential (mV)
 * - I_e: External input current (pA)
 * - Theta_rel: Relative spike threshold (mV)
 * - V_reset_rel: Relative reset potential (mV)
 * - tau_ex: Excitatory PSC time constant (ms)
 * - tau_in: Inhibitory PSC time constant (ms)
 * - t_ref: Refractory period (ms)
 *
 * State Variables:
 * ----------------
 * - V_m_rel: Relative membrane potential
 * - I_syn_ex: Excitatory synaptic current
 * - I_syn_in: Inhibitory synaptic current
 * - refractory_step: Refractory period counter
 *
 * Integration Points:
 * ------------------
 * - Connect: Synaptic inputs via currents
 * - Spike buffer: Output spike delivery
 * - Recording: Multimeter compatibility
 *
 * Performance:
 * ------------
 * - Very fast due to exact integration
 * - Scales efficiently to large populations
 * - Minimal computational overhead
 * - Optimized GPU memory usage
 *
 * Scientific Background:
 * ----------------------
 * Based on classic integrate-and-fire model:
 * - Lapicque (1907): Original leaky integrator
 * - Stein (1965): Stochastic version
 * - Tuckwell (1988): Theory and applications
 *
 * Applications:
 * -------------
 * - Basic spiking neuron dynamics
 * - Network oscillations
 * - Rate coding networks
 * - Educational demonstrations
 * - Large-scale network simulations
 *
 * @see iaf_psc_exp.h Model interface
 * @see propagator_stability.h Exact integration
 * @see nestgpu.h Main simulation engine
 */

// adapted from:
// https://github.com/nest/nest-simulator/blob/master/models/iaf_psc_exp.cpp

#include "iaf_psc_exp.h"
#include "propagator_stability.h"
#include "spike_buffer.h"
#include <cmath>
#include <config.h>
#include <iostream>

using namespace iaf_psc_exp_ns;

extern __constant__ float NESTGPUTimeResolution;
extern __device__ double propagator_32( double, double, double, double );

#define I_syn_ex var[ i_I_syn_ex ]
#define I_syn_in var[ i_I_syn_in ]
#define V_m_rel var[ i_V_m_rel ]
#define refractory_step var[ i_refractory_step ]

#define tau_m param[ i_tau_m ]
#define C_m param[ i_C_m ]
#define E_L param[ i_E_L ]
#define I_e param[ i_I_e ]
#define Theta_rel param[ i_Theta_rel ]
#define V_reset_rel param[ i_V_reset_rel ]
#define tau_ex param[ i_tau_ex ]
#define tau_in param[ i_tau_in ]
// #define rho param[i_rho]
// #define delta param[i_delta]
#define t_ref param[ i_t_ref ]
#define den_delay param[ i_den_delay ]

#define P20 param[ i_P20 ]
#define P11ex param[ i_P11ex ]
#define P11in param[ i_P11in ]
#define P21ex param[ i_P21ex ]
#define P21in param[ i_P21in ]
#define P22 param[ i_P22 ]

__global__ void
iaf_psc_exp_Calibrate( int n_node, float* param_arr, int n_param, float h )
{
  int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_neuron < n_node )
  {
    float* param = param_arr + n_param * i_neuron;

    P11ex = exp( -h / tau_ex );
    P11in = exp( -h / tau_in );
    P22 = exp( -h / tau_m );
    P21ex = ( float ) propagator_32( tau_ex, tau_m, C_m, h );
    P21in = ( float ) propagator_32( tau_in, tau_m, C_m, h );
    P20 = tau_m / C_m * ( 1.0 - P22 );
  }
}

__global__ void
iaf_psc_exp_Update( int n_node, int i_node_0, float* var_arr, float* param_arr, int n_var, int n_param )
{
  int i_neuron = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_neuron < n_node )
  {
    float* var = var_arr + n_var * i_neuron;
    float* param = param_arr + n_param * i_neuron;

    if ( refractory_step > 0.0 )
    {
      // neuron is absolute refractory
      refractory_step -= 1.0;
    }
    else
    { // neuron is not refractory, so evolve V
      V_m_rel = V_m_rel * P22 + I_syn_ex * P21ex + I_syn_in * P21in + I_e * P20;
    }
    // exponential decaying PSCs
    I_syn_ex *= P11ex;
    I_syn_in *= P11in;

    if ( V_m_rel >= Theta_rel )
    { // threshold crossing
      PushSpike( i_node_0 + i_neuron, 1.0 );
      V_m_rel = V_reset_rel;
      refractory_step = ( int ) round( t_ref / NESTGPUTimeResolution );
    }
  }
}

iaf_psc_exp::~iaf_psc_exp()
{
  FreeVarArr();
  FreeParamArr();
}

int
iaf_psc_exp::Init( int i_node_0, int n_node, int /*n_port*/, int i_group )
{
  BaseNeuron::Init( i_node_0, n_node, 2 /*n_port*/, i_group );
  node_type_ = i_iaf_psc_exp_model;

  n_scal_var_ = N_SCAL_VAR;
  n_var_ = n_scal_var_;
  n_scal_param_ = N_SCAL_PARAM;
  n_param_ = n_scal_param_;

  AllocParamArr();
  AllocVarArr();

  scal_var_name_ = iaf_psc_exp_scal_var_name;
  scal_param_name_ = iaf_psc_exp_scal_param_name;

  SetScalParam( 0, n_node, "tau_m", 10.0 );                    // in ms
  SetScalParam( 0, n_node, "C_m", 250.0 );                     // in pF
  SetScalParam( 0, n_node, "E_L", -70.0 );                     // in mV
  SetScalParam( 0, n_node, "I_e", 0.0 );                       // in pA
  SetScalParam( 0, n_node, "Theta_rel", -55.0 - ( -70.0 ) );   // relative to E_L_
  SetScalParam( 0, n_node, "V_reset_rel", -70.0 - ( -70.0 ) ); // relative to E_L_
  SetScalParam( 0, n_node, "tau_ex", 2.0 );                    // in ms
  SetScalParam( 0, n_node, "tau_in", 2.0 );                    // in ms
  // SetScalParam(0, n_node, "rho", 0.01 );             // in 1/s
  // SetScalParam(0, n_node, "delta", 0.0 );            // in mV
  SetScalParam( 0, n_node, "t_ref", 2.0 );     // in ms
  SetScalParam( 0, n_node, "den_delay", 0.0 ); // in ms
  SetScalParam( 0, n_node, "P20", 0.0 );
  SetScalParam( 0, n_node, "P11ex", 0.0 );
  SetScalParam( 0, n_node, "P11in", 0.0 );
  SetScalParam( 0, n_node, "P21ex", 0.0 );
  SetScalParam( 0, n_node, "P21in", 0.0 );
  SetScalParam( 0, n_node, "P22", 0.0 );

  SetScalVar( 0, n_node, "I_syn_ex", 0.0 );
  SetScalVar( 0, n_node, "I_syn_in", 0.0 );
  SetScalVar( 0, n_node, "V_m_rel", -70.0 - ( -70.0 ) ); // in mV, relative to E_L
  SetScalVar( 0, n_node, "refractory_step", 0 );

  // multiplication factor of input signal is always 1 for all nodes
  float input_weight = 1.0;
  CUDAMALLOCCTRL( "&port_weight_arr_", &port_weight_arr_, sizeof( float ) );
  gpuErrchk( cudaMemcpy( port_weight_arr_, &input_weight, sizeof( float ), cudaMemcpyHostToDevice ) );
  port_weight_arr_step_ = 0;
  port_weight_port_step_ = 0;

  // input spike signal is stored in I_syn_ex, I_syn_in
  port_input_arr_ = GetVarArr() + GetScalVarIdx( "I_syn_ex" );
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = 1;

  den_delay_arr_ = GetParamArr() + GetScalParamIdx( "den_delay" );

  return 0;
}

int
iaf_psc_exp::Update( long long it, double t1 )
{
  // std::cout << "iaf_psc_exp neuron update\n";
  iaf_psc_exp_Update<<< ( n_node_ + 1023 ) / 1024, 1024 >>>(
    n_node_, i_node_0_, var_arr_, param_arr_, n_var_, n_param_ );
  // gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

int
iaf_psc_exp::Free()
{
  FreeVarArr();
  FreeParamArr();

  return 0;
}

int
iaf_psc_exp::Calibrate( double, float time_resolution )
{
  iaf_psc_exp_Calibrate<<< ( n_node_ + 1023 ) / 1024, 1024 >>>( n_node_, param_arr_, n_param_, time_resolution );

  return 0;
}
