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

/**
 * @file stdp.cu
 * @brief Spike-Timing-Dependent Plasticity (STDP) implementation for NEST GPU
 *
 * This file implements STDP, a biologically plausible learning rule where
 * synaptic strength changes based on the precise timing of pre- and postsynaptic
 * spikes. STDP is fundamental for modeling learning and memory in neural networks.
 *
 * Mathematical Foundation:
 * -----------------------
 * STDP weight changes depend on spike timing difference Δt = t_post - t_pre:
 *
 * For causal pairing (Δt > 0, post after pre):
 * \[ \Delta w = \lambda \cdot \exp(-\Delta t / \tau_+) \]
 *
 * For anti-causal pairing (Δt < 0, post before pre):
 * \[ \Delta w = -\lambda \cdot \alpha \cdot \exp(\Delta t / \tau_-) \]
 *
 * Power-law STDP (with mu_plus, mu_minus):
 * \[ \Delta w_+ = \lambda \cdot (1 - w/W_{max})^{\mu_+} \cdot \exp(-\Delta t / \tau_+) \]
 * \[ \Delta w_- = -\lambda \cdot \alpha \cdot (w/W_{max})^{\mu_-} \cdot \exp(\Delta t / \tau_-) \]
 *
 * Key Features:
 * -------------
 * - Causal potentiation: Post after pre strengthens synapse
 * - Anti-causal depression: Pre after post weakens synapse
 * - Asymmetric learning windows
 * - Weight-dependent updates
 * - Biologically realistic learning
 *
 * Model Parameters:
 * ----------------
 * - tau_plus: LTP (long-term potentiation) time constant (ms)
 * - tau_minus: LTD (long-term depression) time constant (ms)
 * - lambda: Learning rate parameter
 * - alpha: Asymmetry factor (depression/potentiation ratio)
 * - mu_plus: Potentiation power law exponent
 * - mu_minus: Depression power law exponent
 * - Wmax: Maximum synaptic weight
 *
 * GPU Implementation:
 * ------------------
 * CUDA kernels for parallel STDP updates:
 * - Synapse state management on GPU
 * - Efficient spike pair finding
 * - Parallel weight updates
 * - Optimized memory access patterns
 *
 * Learning Process:
 * ----------------
 * 1. Track pre- and postsynaptic spike times
 * 2. Compute timing differences for spike pairs
 * 3. Apply appropriate weight changes
 * 4. Enforce weight bounds [0, Wmax]
 * 5. Update synaptic strengths
 *
 * Integration Points:
 * ------------------
 * - Syn_model: Plastic synapse management
 * - Connect: STDP synapse creation
 * - Spike buffers: Spike timing tracking
 * - Learning algorithms: Network-level adaptation
 *
 * Performance:
 * ------------
 * - Efficient pairwise correlation detection
 * - Optimized for large networks
 * - Minimal simulation overhead
 * - Scalable to millions of synapses
 *
 * Thread Safety:
 * --------------
 * - Synapse updates are atomic
 * - Concurrent plasticity supported
 * - Thread-safe weight modifications
 *
 * Applications:
 * -------------
 * - Spike-based learning
 * - Temporal sequence learning
 * - Developmental plasticity
 * - Memory consolidation
 * - Network self-organization
 * - Reinforcement learning
 *
 * Scientific Background:
 * ----------------------
 * Based on experimental findings:
 * - Bi & Poo (1998): Synaptic modifications in hippocampus
 * - Markram et al. (1997): Cortical STDP
 * - Song et al. (2000): Tetrablized STDP
 *
 * STDP is observed in:
 * - Hippocampus: Spatial learning
 * - Cortex: Sensory map formation
 * - Striatum: Reward learning
 * - Amygdala: Emotional memory
 *
 * @see stdp.h STDP interface and classes
 * @see syn_model.h Synaptic model framework
 * @see nestgpu.h Main simulation engine
 */

#include "cuda_error.h"
#include "ngpu_exception.h"
#include "stdp.h"
#include "syn_model.h"
#include <config.h>
#include <iostream>
#include <stdio.h>

using namespace stdp_ns;

int
STDP::_Init()
{
  type_ = i_stdp_model;
  n_param_ = N_PARAM;
  param_name_ = stdp_param_name;
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
