/*
 *  aeif_psc_alpha.cu
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
 * @file aeif_psc_alpha.cu
 * @brief Adaptive exponential integrate-and-fire neuron with current-based alpha synapses
 *
 * This file implements the aeif_psc_alpha neuron model, combining
 * exponential integrate-and-fire dynamics with adaptation current and
 * current-based synaptic inputs with alpha-function postsynaptic currents.
 *
 * Mathematical Model:
 * -------------------
 * The aeif_psc_alpha model extends standard integrate-and-fire:
 *
 * Membrane potential dynamics:
 * \[ C_m \frac{dV}{dt} = -g_L(V-E_L) + g_L \Delta_T \exp\left(\frac{V-V_{th}}{\Delta_T}\right) - w + I_{syn} + I_e \]
 *
 * Adaptation current dynamics:
 * \[ \tau_w \frac{dw}{dt} = a(V-E_L) - w \]
 *
 * Alpha function PSC:
 * The alpha function provides realistic synaptic dynamics:
 * \[ I_{syn}(t) = \frac{t}{\tau_{syn}} \exp(1 - t/\tau_{syn}) \]
 *
 * Key Features:
 * -------------
 * - Exponential approach to spike threshold
 * - Spike-triggered adaptation current
 * - Current-based synaptic inputs
 * - Alpha-function postsynaptic currents
 * - Realistic synaptic dynamics
 *
 * GPU Implementation:
 * ------------------
 * - RK5 (Runge-Kutta 5th order) integration
 * - Parallel updates across populations
 * - Efficient memory access patterns
 * - Spike detection on GPU
 *
 * Integration Method:
 * ------------------
 * Uses RK5 for accurate integration:
 * - Adaptive stepsize control
 * - High accuracy for stiff equations
 * - Stable for physiological parameters
 *
 * Model Parameters:
 * ----------------
 * - C_m: Membrane capacitance (pF)
 * - g_L: Membrane conductance (nS)
 * - E_L: Resting potential (mV)
 * - V_th: Spike threshold (mV)
 * - Delta_T: Slope factor (mV)
 * - a, b: Adaptation parameters
 * - tau_w: Adaptation time constant (ms)
 * - tau_syn: Synaptic time constant (ms)
 *
 * Integration Points:
 * ------------------
 * - Connect: Synaptic inputs via currents
 * - Spike buffer: Output spike delivery
 * - Recording: Multimeter compatibility
 *
 * @see aeif_psc_alpha.h Model interface
 * @see aeif_psc_alpha_kernel.h GPU kernels
 * @see rk5.h Integration method
 */

#include "aeif_psc_alpha.h"
#include "aeif_psc_alpha_kernel.h"
#include "rk5.h"
#include <cmath>
#include <config.h>
#include <iostream>

namespace aeif_psc_alpha_ns
{

__device__ void
NodeInit( int n_var, int n_param, double x, float* y, float* param, aeif_psc_alpha_rk5 data_struct )
{
  // int array_idx = threadIdx.x + blockIdx.x * blockDim.x;

  V_th = -50.4;
  Delta_T = 2.0;
  g_L = 30.0;
  E_L = -70.6;
  C_m = 281.0;
  a = 4.0;
  b = 80.5;
  tau_w = 144.0;
  I_e = 0.0;
  V_peak = 0.0;
  V_reset = -60.0;
  t_ref = 0.0;
  den_delay = 0.0;

  V_m = E_L;
  w = 0.0;
  refractory_step = 0;
  I_syn_ex = 0.0;
  I_syn_in = 0.0;
  I1_syn_ex = 0.0;
  I1_syn_in = 0.0;
  tau_syn_ex = 0.2;
  tau_syn_in = 2.0;
}

__device__ void
NodeCalibrate( int n_var, int n_param, double x, float* y, float* param, aeif_psc_alpha_rk5 data_struct )
{
  // int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  refractory_step = 0;
  // set the right threshold depending on Delta_T
  if ( Delta_T <= 0.0 )
  {
    V_peak = V_th; // same as IAF dynamics for spikes if Delta_T == 0.
  }
  I0_ex = M_E / tau_syn_ex;
  I0_in = M_E / tau_syn_in;
}

} // namespace aeif_psc_alpha_ns

__device__ void
NodeInit( int n_var, int n_param, double x, float* y, float* param, aeif_psc_alpha_rk5 data_struct )
{
  aeif_psc_alpha_ns::NodeInit( n_var, n_param, x, y, param, data_struct );
}

__device__ void
NodeCalibrate( int n_var, int n_param, double x, float* y, float* param, aeif_psc_alpha_rk5 data_struct )

{
  aeif_psc_alpha_ns::NodeCalibrate( n_var, n_param, x, y, param, data_struct );
}

using namespace aeif_psc_alpha_ns;

int
aeif_psc_alpha::Init( int i_node_0, int n_node, int n_port, int i_group )
{
  BaseNeuron::Init( i_node_0, n_node, n_port, i_group );
  node_type_ = i_aeif_psc_alpha_model;
  n_scal_var_ = N_SCAL_VAR;
  n_scal_param_ = N_SCAL_PARAM;
  n_group_param_ = N_GROUP_PARAM;

  n_var_ = n_scal_var_;
  n_param_ = n_scal_param_;

  group_param_ = new float[ N_GROUP_PARAM ];

  scal_var_name_ = aeif_psc_alpha_scal_var_name;
  scal_param_name_ = aeif_psc_alpha_scal_param_name;
  group_param_name_ = aeif_psc_alpha_group_param_name;
  // rk5_data_struct_.node_type_ = i_aeif_psc_alpha_model;
  rk5_data_struct_.i_node_0_ = i_node_0_;

  SetGroupParam( "h_min_rel", 1.0e-3 );
  SetGroupParam( "h0_rel", 1.0e-2 );
  h_ = h0_rel_ * 0.1;

  rk5_.Init( n_node, n_var_, n_param_, 0.0, h_, rk5_data_struct_ );
  var_arr_ = rk5_.GetYArr();
  param_arr_ = rk5_.GetParamArr();

  port_weight_arr_ = GetParamArr() + GetScalParamIdx( "I0_ex" );
  port_weight_arr_step_ = n_param_;
  port_weight_port_step_ = 1;

  port_input_arr_ = GetVarArr() + GetScalVarIdx( "I1_syn_ex" );
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = 1;
  den_delay_arr_ = GetParamArr() + GetScalParamIdx( "den_delay" );

  return 0;
}

int
aeif_psc_alpha::Calibrate( double time_min, float time_resolution )
{
  h_min_ = h_min_rel_ * time_resolution;
  h_ = h0_rel_ * time_resolution;
  rk5_.Calibrate( time_min, h_, rk5_data_struct_ );

  return 0;
}

int
aeif_psc_alpha::Update( long long it, double t1 )
{
  rk5_.Update< N_SCAL_VAR, N_SCAL_PARAM >( t1, h_min_, rk5_data_struct_ );

  return 0;
}
