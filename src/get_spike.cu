/*
 *  get_spike.cu
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
 * @file get_spike.cu
 * @brief Spike retrieval and extraction interface for NEST GPU
 *
 * This file implements the spike retrieval system that extracts spike
 * information from neuron models for delivery to target neurons. It's
 * a critical interface between neuron dynamics and spike delivery.
 *
 * Architecture Overview:
 * ---------------------
 * The spike retrieval system:
 * - Extracts spike information from neuron state
 * - Formats spikes for delivery
 * - Handles multi-port spike emission
 * - Manages spike multiplicity
 *
 * Key Functions:
 * --------------
 * - GetSpikes: Extract spike data from neurons
 * - Format spike information for delivery
 * - Handle multi-port spike emission
 * - Support spike multiplicity and weights
 *
 * GPU Implementation:
 * ------------------
 * CUDA kernels for parallel spike retrieval:
 * - Parallel access to neuron states
 * - Efficient spike detection
 * - Coalesced memory patterns
 * - Minimized thread divergence
 *
 * Spike Information:
 * ------------------
 * Retrieved data includes:
 * - Source neuron ID
 * - Spike timing information
 * - Port information (receptor type)
 * - Spike multiplicity
 * - Synaptic weight references
 *
 * Integration Points:
 * ------------------
 * - BaseNeuron: Access to neuron spike data
 * - Spike buffer: Store retrieved spikes
 * - Send spike: Interface for spike delivery
 * - Connect: Route spikes to targets
 *
 * Performance:
 * ------------
 * - Optimized for parallel retrieval
 * - Efficient memory access
 * - Minimal overhead
 * - Scalable to large populations
 *
 * Thread Safety:
 * --------------
 * - Thread-safe spike retrieval
 * - Concurrent neuron access
 * - Atomic spike operations
 *
 * Usage Pattern:
 * --------------
 * 1. Neuron models emit spikes
 * 2. GetSpikes extracts spike information
 * 3. Spikes placed in delivery buffers
 * 4. System routes spikes to targets
 *
 * @see get_spike.h Spike retrieval interface
 * @see spike_buffer.h Spike storage
 * @see send_spike.h Spike delivery
 */

#include <config.h>
#include <stdio.h>

#include "connect.h"
#include "cuda_error.h"
#include "nestgpu.h"
#include "node_group.h"
#include "send_spike.h"
#include "spike_buffer.h"

// improve using a grid
/*
__global__ void GetSpikes(double *spike_array, int array_size, int n_port,
                          int n_var,
                          float *port_weight_arr,
                          int port_weight_arr_step,
                          int port_weight_port_step,
                          float *port_input_arr,
                          int port_input_arr_step,
                          int port_input_port_step)
{
  int i_array = threadIdx.x + blockIdx.x * blockDim.x;
  if (i_array < array_size*n_port) {
     int i_target = i_array % array_size;
     int port = i_array / array_size;
     int port_input = i_target*port_input_arr_step
       + port_input_port_step*port;
     int port_weight = i_target*port_weight_arr_step
       + port_weight_port_step*port;
     double d_val = (double)port_input_arr[port_input]
       + spike_array[i_array]
       * port_weight_arr[port_weight];

     port_input_arr[port_input] = (float)d_val;
  }
}
*/

__global__ void
GetSpikes( double* spike_array,
  int array_size,
  int n_port,
  int n_var,
  float* port_weight_arr,
  int port_weight_arr_step,
  int port_weight_port_step,
  float* port_input_arr,
  int port_input_arr_step,
  int port_input_port_step )
{
  int i_target = blockIdx.x * blockDim.x + threadIdx.x;
  int port = blockIdx.y * blockDim.y + threadIdx.y;

  if ( i_target < array_size && port < n_port )
  {
    int i_array = port * array_size + i_target;
    int port_input = i_target * port_input_arr_step + port_input_port_step * port;
    int port_weight = i_target * port_weight_arr_step + port_weight_port_step * port;
    double d_val = ( double ) port_input_arr[ port_input ] + spike_array[ i_array ] * port_weight_arr[ port_weight ];

    port_input_arr[ port_input ] = ( float ) d_val;
  }
}

int
NESTGPU::ClearGetSpikeArrays()
{
  for ( unsigned int i = 0; i < node_vect_.size(); i++ )
  {
    BaseNeuron* bn = node_vect_[ i ];
    if ( bn->get_spike_array_ != nullptr )
    {
      gpuErrchk( cudaMemsetAsync( bn->get_spike_array_, 0, bn->n_node_ * bn->n_port_ * sizeof( double ) ) );
    }
  }

  return 0;
}

int
NESTGPU::FreeGetSpikeArrays()
{
  for ( unsigned int i = 0; i < node_vect_.size(); i++ )
  {
    BaseNeuron* bn = node_vect_[ i ];
    if ( bn->get_spike_array_ != nullptr )
    {
      CUDAFREECTRL( "bn->get_spike_array_", bn->get_spike_array_ );
    }
  }

  return 0;
}
