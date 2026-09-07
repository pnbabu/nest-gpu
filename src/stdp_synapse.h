/*
 *  stdp_synapse.h
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

#ifndef STDP_SYNAPSE_H
#define STDP_SYNAPSE_H
#ifdef HAVE_SYN_STATE_VARS
#include <cmath>
#include "syn_model.h"

/* BeginUserDocs: synapse, spike-timing-dependent plasticity

Short description
+++++++++++++++++

Synapse type for spike-timing dependent plasticity

Description
+++++++++++

The STDP class is a type of synapse model used to create
synapses that enable spike timing dependent plasticity
(as defined in [1]_).
Here the weight dependence exponent can be set separately
for potentiation and depression.


Parameters
++++++++++

========== =======  ======================================================
 tau_plus  ms       Time constant of STDP window, potentiation
 tau_minus ms       Time constant of STDP window, depression
 lambda    real     Step size
 alpha     real     Asymmetry parameter (scales depression increments as
                    alpha*lambda)
 mu_plus   real     Weight dependence exponent, potentiation
 mu_minus  real     Weight dependence exponent, depression
 Wmax      real     Maximum allowed weight
========== =======  ======================================================


References
++++++++++

.. [1] Guetig et al. (2003). Learning input correlations through nonlinear
       temporally asymmetric hebbian plasticity. Journal of Neuroscience,
       23:3697-3714 DOI: https://doi.org/10.1523/JNEUROSCI.23-09-03697.2003


EndUserDocs */

extern __device__ float* ConnectionStateVars;

namespace stdp_synapse_ns
{
enum ParamIndexes
{
  i_tau_plus = 0,
  i_tau_minus,
  i_lambda,
  i_alpha,
  i_mu_plus,
  i_mu_minus,
  i_Wmax, // i_den_delay,
  N_PARAM
};

const std::string stdp_synapse_param_name[ N_PARAM ] = {
  "tau_plus",
  "tau_minus",
  "lambda",
  "alpha",
  "mu_plus",
  "mu_minus",
  "Wmax"
  //, "den_delay"
};

enum StateIndexes
{
    i_state_w = 0,           // Weight state variable
    i_state_pre_trace,        // Presynaptic trace
    i_state_post_trace,       // Postsynaptic trace
    N_STATE_VARS              // STDP has 3 state variables
};

const std::string stdp_synapse_state_name[N_STATE_VARS] = {
    "w", "pre_trace", "post_trace"
};

// Definitions of the device functions below are in stdp_synapse.cu
__device__ void STDPSynapsePreTraceUpdate( int i_conn );

__device__ void STDPSynapsePostTraceUpdate( int i_conn );

__device__ void STDPSynapseUpdate( float* weight_pt, float Dt, float* param, int i_conn );
} // namespace stdp_synapse_ns

class STDPSynapse : public SynModel
{
  int _Init();

public:
  STDPSynapse()
  {
    _Init();
  }

  int
  Init()
  {
    return _Init();
  }
};

#endif
#endif
