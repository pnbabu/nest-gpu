/*
 *  izhikevich.cu
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
 * @file izhikevich.cu
 * @brief Izhikevich neuron model implementation
 *
 * This file implements the Izhikevich neuron model, a simple
 * spiking neuron model that can reproduce various firing patterns
 * while remaining computationally efficient.
 *
 * Mathematical Model:
 * -------------------
 * The Izhikevich model consists of two differential equations:
 *
 * \[ \frac{dV}{dt} = 0.04V^2 + 5V + 140 - u + I \]
 * \[ \frac{du}{dt} = a(bV - u) \]
 *
 * After-spike reset:
 * \[ \text{if } V \geq 30 \text{ mV:} \]
 * \[ V \leftarrow c, \quad u \leftarrow u + d \]
 *
 * Where:
 * - V: Membrane potential (mV)
 * - u: Recovery variable (represents membrane potential recovery)
 * - I: Synaptic input current
 * - a, b, c, d: Model parameters
 *
 * Key Features:
 * -------------
 * - Computationally efficient (simple quadratic equation)
 * - Can reproduce various firing patterns:
 *   - Regular spiking (RS)
 *   - Intrinsically bursting (IB)
 *   - Chattering (CH)
 *   - Fast spiking (FS)
 *   - Thalamo-cortical (TC)
 *   - Resonator (RZ)
 *   - Low-threshold spiking (LTS)
 *
 * Parameter Regimes:
 * -----------------
 * Different firing patterns achieved with parameter sets:
 * - Regular spiking: (a,b,c,d) = (0.02, 0.2, -65, 8)
 * - Intrinsically bursting: (a,b,c,d) = (0.02, 0.2, -55, 4)
 * - Fast spiking: (a,b,c,d) = (0.1, 0.2, -65, 2)
 *
 * GPU Implementation:
 * ------------------
 * CUDA kernel for parallel updates:
 * - Each thread updates one neuron
 * - Coalesced memory access patterns
 * - Efficient spike detection
 * - Minimal thread divergence
 *
 * State Variables:
 * ----------------
 * - V_m: Membrane potential (mV)
 * - u: Recovery variable
 * - I_syn: Synaptic current
 * - refractory_step: Refractory counter
 *
 * Model Parameters:
 * ----------------
 * - V_th: Spike threshold (typically 30 mV)
 * - a: Time scale of recovery variable
 * - b: Sensitivity of recovery variable
 * - c: After-spike reset value for V
 * - d: After-spike reset value for u
 * - t_ref: Refractory period (ms)
 * - I_e: External input current
 *
 * Integration Method:
 * ------------------
 * Uses Euler method with small time steps:
 * - Simple and fast
 * - Sufficient for this model
 * - GPU-friendly implementation
 *
 * Performance:
 * ------------
 * - Very fast due to simple equations
 * - Scales efficiently to large populations
 * - Minimal computational overhead
 *
 * Integration Points:
 * ------------------
 * - Connect: Current-based synaptic inputs
 * - Spike buffer: Output spike delivery
 * - Recording: Compatible with multimeter
 *
 * Scientific Background:
 * ----------------------
 * Based on Izhikevich (2003):
 * "Simple Model of Spiking Neurons"
 * IEEE Transactions on Neural Networks, 14(6):1569-1572
 *
 * The model was designed to balance biological realism
 * with computational efficiency, making it ideal for
 * large-scale network simulations.
 *
 * @see izhikevich.h Model interface
 * @see nestgpu.h Main simulation engine
 */

#include "izhikevich.h"
#include "spike_buffer.h"
#include <cmath>
#include <config.h>
#include <iostream>

using namespace izhikevich_ns;

extern __constant__ float NESTGPUTimeResolution;

#define I_syn var[ i_I_syn ]
#define V_m var[ i_V_m ]
#define u var[ i_u ]
#define refractory_step var[ i_refractory_step ]
#define I_e param[ i_I_e ]
#define den_delay param[ i_den_delay ]

#define V_th_ group_param_[ i_V_th ]
#define a_ group_param_[ i_a ]
#define b_ group_param_[ i_b ]
#define c_ group_param_[ i_c ]
#define d_ group_param_[ i_d ]
#define t_ref_ group_param_[ i_t_ref ]

__global__ void
izhikevich_Update( int n_node,
  int i_node_0,
  float* var_arr,
  float* param_arr,
  int n_var,
  int n_param,
  float V_th,
  float a,
  float b,
  float c,
  float d,
  int n_refractory_steps,
  float h )
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
    { // neuron is not refractory, so evolve V and u
      float v_old = V_m;
      float u_old = u;

      V_m += h * ( 0.04 * v_old * v_old + 5.0 * v_old + 140.0 - u_old + I_e ) + I_syn;
      u += h * a * ( b * v_old - u_old );
    }
    I_syn = 0;

    if ( V_m >= V_th )
    { // send spike
      PushSpike( i_node_0 + i_neuron, 1.0 );
      V_m = c;
      u += d; // spike-driven adaptation
      refractory_step = n_refractory_steps;
      if ( refractory_step < 0 )
      {
        refractory_step = 0;
      }
    }
  }
}

izhikevich::~izhikevich()
{
  FreeVarArr();
  FreeParamArr();
}

int
izhikevich::Init( int i_node_0, int n_node, int /*n_port*/, int i_group )
{
  BaseNeuron::Init( i_node_0, n_node, 1 /*n_port*/, i_group );
  node_type_ = i_izhikevich_model;

  n_scal_var_ = N_SCAL_VAR;
  n_var_ = n_scal_var_;
  n_scal_param_ = N_SCAL_PARAM;
  n_group_param_ = N_GROUP_PARAM;
  n_param_ = n_scal_param_;

  AllocParamArr();
  AllocVarArr();
  group_param_ = new float[ N_GROUP_PARAM ];

  scal_var_name_ = izhikevich_scal_var_name;
  scal_param_name_ = izhikevich_scal_param_name;
  group_param_name_ = izhikevich_group_param_name;

  SetScalParam( 0, n_node, "I_e", 0.0 );       // in pA
  SetScalParam( 0, n_node, "den_delay", 0.0 ); // in ms

  SetScalVar( 0, n_node, "I_syn", 0.0 );
  SetScalVar( 0, n_node, "V_m", -70.0 ); // in mV
  SetScalVar( 0, n_node, "u", -70.0 * 0.2 );
  SetScalVar( 0, n_node, "refractory_step", 0 );

  SetGroupParam( "V_th", 30.0 );
  SetGroupParam( "a", 0.02 );
  SetGroupParam( "b", 0.2 );
  SetGroupParam( "c", -65.0 );
  SetGroupParam( "d", 8.0 );
  SetGroupParam( "t_ref", 0.0 );

  // multiplication factor of input signal is always 1 for all nodes
  float input_weight = 1.0;
  CUDAMALLOCCTRL( "&port_weight_arr_", &port_weight_arr_, sizeof( float ) );
  gpuErrchk( cudaMemcpy( port_weight_arr_, &input_weight, sizeof( float ), cudaMemcpyHostToDevice ) );
  port_weight_arr_step_ = 0;
  port_weight_port_step_ = 0;

  // input spike signal is stored in I_syn
  port_input_arr_ = GetVarArr() + GetScalVarIdx( "I_syn" );
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = 0;

  return 0;
}

int
izhikevich::Update( long long it, double t1 )
{
  // std::cout << "izhikevich neuron update\n";
  float h = time_resolution_;
  int n_refractory_steps = int( round( t_ref_ / h ) );

  izhikevich_Update<<< ( n_node_ + 1023 ) / 1024, 1024 >>>(
    n_node_, i_node_0_, var_arr_, param_arr_, n_var_, n_param_, V_th_, a_, b_, c_, d_, n_refractory_steps, h );
  // gpuErrchk( cudaDeviceSynchronize() );

  return 0;
}

int
izhikevich::Free()
{
  FreeVarArr();
  FreeParamArr();
  delete[] group_param_;

  return 0;
}
