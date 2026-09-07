/*
 *  stdp.cu
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
#include <config.h>
#ifdef HAVE_SYN_STATE_VARS
#include "cuda_error.h"
#include "ngpu_exception.h"
#include "stdp_synapse.h"
#include <iostream>
#include <stdio.h>

namespace stdp_synapse_ns
{

__device__ void
update_internal_state( float Dt, float* param, int i_conn )
{
  // Update the trace values
  int base_idx = N_STATE_VARS * i_conn;
  const double pre_trace_tmp = exp( -( double ) Dt / param[ i_tau_plus ] ) * ConnectionStateVars[ base_idx + i_state_pre_trace ];
  const double post_trace_tmp = exp( ( double ) Dt / param[ i_tau_minus ] ) * ConnectionStateVars[ base_idx + i_state_post_trace ];
  ConnectionStateVars[ base_idx + i_state_pre_trace ] = pre_trace_tmp;
  ConnectionStateVars[ base_idx + i_state_post_trace ] = post_trace_tmp;
}

__device__ void
STDPSynapsePreTraceUpdate( int i_conn )
{
  int base_idx = N_STATE_VARS * i_conn;
  ConnectionStateVars[ base_idx + i_state_pre_trace ] = 1.0f;
}

__device__ void
STDPSynapsePostTraceUpdate( int i_conn )
{
  int base_idx = N_STATE_VARS * i_conn;
  ConnectionStateVars[ base_idx + i_state_post_trace ] = 1.0f;
}

__device__ void
STDPSynapseUpdate( float* weight_pt, float Dt, float* param, int i_conn )
{
  // printf("In STDPSynapseUpdate function. Dt: %f\n", Dt);
  double lambda = param[ i_lambda ];
  double alpha = param[ i_alpha ];
  double mu_plus = param[ i_mu_plus ];
  double mu_minus = param[ i_mu_minus ];
  double Wmax = param[ i_Wmax ];
  // double den_delay = param[i_den_delay];

  // State vars
  int base_idx = N_STATE_VARS * i_conn;

  ConnectionStateVars[ base_idx + i_state_w ] = *weight_pt;
  double w = *weight_pt;
  double w1;

  update_internal_state( Dt, param, i_conn );
  // Dt += den_delay;
  if ( Dt >= 0 )
  {
    // facilitation
    double pre_trace = ConnectionStateVars[ base_idx + i_state_pre_trace ];
    // printf("pre_trace: %f\n", pre_trace);
    w1 = Wmax * ( w / Wmax + ( lambda * pow( ( 1. - ( w / Wmax ) ), mu_plus ) * pre_trace ) );
  }
  else
  {
    // depression
    double post_trace = ConnectionStateVars[ base_idx + i_state_post_trace ];
    // printf("post_trace: %f\n", post_trace);
    w1 = Wmax * ( w / Wmax - ( alpha * lambda * pow( ( w / Wmax ), mu_minus ) * post_trace ) );
  }

  w1 = w1 > 0.0 ? w1 : 0.0;
  w1 = w1 < Wmax ? w1 : Wmax;
  *weight_pt = ( float ) w1;
  ConnectionStateVars[ base_idx + i_state_w ] = ( float ) w1;
}

} // namespace stdp_synapse_ns

using namespace stdp_synapse_ns;

int
STDPSynapse::_Init()
{
  type_ = i_stdp_synapse_model;
  n_param_ = N_PARAM;
  param_name_ = stdp_synapse_param_name;

  // state vars
  n_state_vars_ = N_STATE_VARS;
  state_name_ = stdp_synapse_state_name;

  CUDAMALLOCCTRL( "&d_param_arr_", &d_param_arr_, n_param_ * sizeof( float ) );
  SetParam( "tau_plus", 20.0 );
  SetParam( "tau_minus", 20.0 );
  SetParam( "lambda", 1.0e-4 );
  SetParam( "alpha", 1.0 );
  SetParam( "mu_plus", 1.0 );
  SetParam( "mu_minus", 1.0 );
  SetParam( "Wmax", 100.0 );
  // SetParam("den_delay", 0.0);

  return 0;
}

// SynModel*
// CreateSTDPSynapse()
// {
//   return new STDPSynapse;
// }
#endif
