/*
 *  rev_spike.cu
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
 * @file rev_spike.cu
 * @brief Reverse connection spike delivery and STDP support for NEST GPU
 *
 * This file implements reverse connection spike delivery, which is
 * essential for spike-timing-dependent plasticity (STDP) and other
 * learning rules that require postsynaptic spike information.
 *
 * Architecture Overview:
 * ---------------------
 * The reverse spike system:
 * - Maintains reverse connection indices
 * - Delivers postsynaptic spikes backwards
 * - Enables STDP pair detection
 * - Supports plasticity updates
 * - Efficient backward communication
 *
 * Key Components:
 * ---------------
 * - RevSpikeNum: Reverse spike counts
 * - RevSpikeTarget: Reverse target indices
 * - RevSpikeNConn: Reverse connection counts
 * - RevConnections: Reverse connection arrays
 * - TargetRevConnection: Per-target reverse indices
 *
 * STDP Support:
 * -------------
 * Reverse connections enable:
 * - Postsynaptic spike detection
 * - Pre-post spike pair identification
 * - Timing difference computation
 * - Weight update calculation
 *
 * GPU Implementation:
 * ------------------
 * CUDA kernels for reverse spike delivery:
 * - Parallel backward spike routing
 * - Efficient connection lookup
 * - Coalesced memory access
 * - Atomic operations for consistency
 *
 * Spike Delivery Process:
 * ----------------------
 * 1. Postsynaptic neuron emits spike
 * 2. Reverse spike delivered to presynaptic neurons
 * 3. Spike timing information recorded
 * 4. STDP pair detection performed
 * 5. Weight updates applied
 *
 * Integration Points:
 * ------------------
 * - Connect: Reverse connection establishment
 * - Syn_model: Plasticity updates
 * - STDP: Timing-based learning
 * - Spike buffer: Spike management
 *
 * Performance:
 * ------------
 * - Efficient backward routing
 * - Optimized memory access
 * - Minimal STDP overhead
 * - Scalable to large networks
 *
 * Thread Safety:
 * --------------
 * - Atomic reverse spike operations
 * - Thread-safe weight updates
 * - Consistent pair detection
 *
 * Usage Pattern:
 * --------------
 * 1. Postsynaptic neuron spikes
 * 2. Reverse spike delivered
 * 3. Presynaptic side receives timing
 * 4. STDP pair identified
 * 5. Weight updated accordingly
 *
 * Scientific Background:
 * ----------------------
 * Reverse spike delivery enables:
 * - Bi-directional STDP
 * - Triplet STDP
 * - Neuromodulated plasticity
 * - Complex learning rules
 *
 * @see rev_spike.h Reverse spike interface
 * @see stdp.h STDP implementation
 * @see syn_model.h Plasticity framework
 */

#include "connect.h"
#include "cuda_error.h"
#include "spike_buffer.h"
#include "syn_model.h"
#include <config.h>
#include <cub/cub.cuh>
#include <stdio.h>

#define SPIKE_TIME_DIFF_GUARD 15000 // must be less than 16384
#define SPIKE_TIME_DIFF_THR 10000   // must be less than GUARD

extern __constant__ long long NESTGPUTimeIdx;

extern __constant__ float NESTGPUTimeResolution;

extern __device__ void SynapseUpdate( int syn_group, float* w, float Dt );

__device__ unsigned int* RevSpikeNum;

__device__ unsigned int* RevSpikeTarget;

__device__ int* RevSpikeNConn;

__device__ int64_t* RevConnections;

__device__ int* TargetRevConnectionSize;

__device__ int64_t** TargetRevConnection;

__global__ void
setTargetRevConnectionsPtKernel( int n_spike_buffer,
  int64_t* target_rev_connection_cumul,
  int64_t** target_rev_connection,
  int64_t* rev_connections )
{
  int i_target = blockIdx.x * blockDim.x + threadIdx.x;
  if ( i_target >= n_spike_buffer )
  {
    return;
  }
  target_rev_connection[ i_target ] = rev_connections + target_rev_connection_cumul[ i_target ];
}

__global__ void
revConnectionInitKernel( int64_t* rev_conn, int* target_rev_conn_size, int64_t** target_rev_conn )
{
  RevConnections = rev_conn;
  TargetRevConnectionSize = target_rev_conn_size;
  TargetRevConnection = target_rev_conn;
}

__global__ void
revSpikeBufferUpdate( unsigned int n_node )
{
  unsigned int i_node = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_node >= n_node )
  {
    return;
  }
  long long target_spike_time_idx = LastRevSpikeTimeIdx[ i_node ];
  // Check if a spike reached the input synapses now
  if ( target_spike_time_idx != NESTGPUTimeIdx )
  {
    return;
  }
  int n_conn = TargetRevConnectionSize[ i_node ];
  if ( n_conn > 0 )
  {
    unsigned int pos = atomicAdd( RevSpikeNum, 1 );
    RevSpikeTarget[ pos ] = i_node;
    RevSpikeNConn[ pos ] = n_conn;
  }
}

__global__ void
setConnectionSpikeTime( unsigned int n_conn, unsigned short time_idx )
{
  unsigned int i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_conn >= n_conn )
  {
    return;
  }
  ConnectionSpikeTime[ i_conn ] = time_idx;
}

__global__ void
resetConnectionSpikeTimeUpKernel( unsigned int n_conn )
{
  unsigned int i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_conn >= n_conn )
  {
    return;
  }
  unsigned short spike_time = ConnectionSpikeTime[ i_conn ];
  if ( spike_time >= 0x8000 )
  {
    ConnectionSpikeTime[ i_conn ] = 0;
  }
}

__global__ void
resetConnectionSpikeTimeDownKernel( unsigned int n_conn )
{
  unsigned int i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_conn >= n_conn )
  {
    return;
  }
  unsigned short spike_time = ConnectionSpikeTime[ i_conn ];
  if ( spike_time < 0x8000 )
  {
    ConnectionSpikeTime[ i_conn ] = 0x8000;
  }
}

__global__ void
deviceRevSpikeInit( unsigned int* rev_spike_num, unsigned int* rev_spike_target, int* rev_spike_n_conn )
{
  RevSpikeNum = rev_spike_num;
  RevSpikeTarget = rev_spike_target;
  RevSpikeNConn = rev_spike_n_conn;
  *RevSpikeNum = 0;
}

__global__ void
revSpikeReset()
{
  *RevSpikeNum = 0;
}
