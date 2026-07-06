/*
 *  send_spike.cu
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
 * @file send_spike.cu
 * @brief Spike emission and delivery system for NEST GPU
 *
 * This file implements the spike emission and delivery system that
 * manages spike propagation from source neurons to target neurons
 * through synaptic connections.
 *
 * Architecture Overview:
 * ---------------------
 * The spike delivery system:
 * - Collects spikes from source neurons
 * - Routes spikes through connections
 * - Delivers spikes to target neurons
 * - Manages spike multiplicity
 * - Handles spike delays
 *
 * Key Components:
 * ---------------
 * - SpikeNum: Count of emitted spikes
 * - SpikeSourceIdx: Source neuron indices
 * - SpikeConnIdx: Connection indices
 * - SpikeMul: Spike multiplicities
 * - SpikeTargetNum: Target counts
 *
 * GPU Implementation:
 * ------------------
 * CUDA kernels for spike handling:
 * - Parallel spike collection
 * - Atomic operations for consistency
 * - Efficient memory access patterns
 * - Minimized thread divergence
 *
 * Spike Delivery Process:
 * ----------------------
 * 1. Neurons emit spikes
 * 2. SendSpike collects spike data
 * 3. Spikes routed through connections
 * 4. Delivered to input spike buffers
 * 5. Target neurons receive inputs
 *
 * Key Functions:
 * -------------
 * - SendSpike: Emit spike with metadata
 * - atomicAdd: Thread-safe spike counting
 * - Spike routing through connections
 * - Delay-based spike scheduling
 *
 * Integration Points:
 * ------------------
 * - Input spike buffer: Receives delivered spikes
 * - Connect: Connection information for routing
 * - Neuron models: Spike emission
 * - Spike buffer: Spike storage
 *
 * Performance:
 * ------------
 * - Atomic operations for thread safety
 * - Coalesced memory writes
 * - Efficient spike routing
 * - Minimal overhead
 *
 * Thread Safety:
 * --------------
 * - Atomic spike emission
 * - Thread-safe counters
 * - Concurrent spike delivery
 *
 * Usage Pattern:
 * --------------
 * 1. Neuron emits spike
 * 2. SendSpike called with spike data
 * 3. Spike routed to connections
 * 4. Delivered to input buffers
 * 5. Processed by target neurons
 *
 * @see send_spike.h Spike emission interface
 * @see input_spike_buffer.cu Spike reception
 * @see connect.h Connection routing
 */

#include "cuda_error.h"
#include "input_spike_buffer.h"
#include "send_spike.h"
#include <config.h>
#include <stdio.h>

int* d_SpikeNum;
int* d_SpikeSourceIdx;
int* d_SpikeConnIdx;
float* d_SpikeMul;
int* d_SpikeTargetNum;

__device__ int MaxSpikeNum;
__device__ int* SpikeNum;
__device__ int* SpikeSourceIdx;
__device__ int* SpikeConnIdx;
__device__ float* SpikeMul;
__device__ int* SpikeTargetNum;

__device__ void
SendSpike( int i_source, int i_conn, float mul, int target_num )
{
  int pos = atomicAdd( SpikeNum, 1 );
  if ( pos >= MaxSpikeNum )
  {
    printf( "Number of spikes larger than MaxSpikeNum: %d\n", MaxSpikeNum );
    *SpikeNum = MaxSpikeNum;
    return;
  }
  SpikeSourceIdx[ pos ] = i_source;
  SpikeConnIdx[ pos ] = i_conn;
  SpikeMul[ pos ] = mul;
  SpikeTargetNum[ pos ] = target_num;
}

__global__ void
DeviceSpikeInit( int* spike_num,
  int* spike_source_idx,
  int* spike_conn_idx,
  float* spike_mul,
  int* spike_target_num,
  int max_spike_num )
{
  SpikeNum = spike_num;
  SpikeSourceIdx = spike_source_idx;
  SpikeConnIdx = spike_conn_idx;
  SpikeMul = spike_mul;
  SpikeTargetNum = spike_target_num;
  MaxSpikeNum = max_spike_num;
  *SpikeNum = 0;
}

void
SpikeInit( int max_spike_num )
{
  // h_SpikeTargetNum = new int[PrefixScan::AllocSize];

  CUDAMALLOCCTRL( "&d_SpikeNum", &d_SpikeNum, sizeof( int ) );
  CUDAMALLOCCTRL( "&d_SpikeSourceIdx", &d_SpikeSourceIdx, max_spike_num * sizeof( int ) );
  CUDAMALLOCCTRL( "&d_SpikeConnIdx", &d_SpikeConnIdx, max_spike_num * sizeof( int ) );
  CUDAMALLOCCTRL( "&d_SpikeMul", &d_SpikeMul, max_spike_num * sizeof( float ) );
  CUDAMALLOCCTRL( "&d_SpikeTargetNum", &d_SpikeTargetNum, max_spike_num * sizeof( int ) );
  // printf("here: SpikeTargetNum size: %d", max_spike_num);
  DeviceSpikeInit<<< 1, 1 >>>(
    d_SpikeNum, d_SpikeSourceIdx, d_SpikeConnIdx, d_SpikeMul, d_SpikeTargetNum, max_spike_num );
  gpuErrchk( cudaPeekAtLastError() );
}

__global__ void
SpikeReset()
{
  if ( input_spike_buffer_ns::algo_ == INPUT_SPIKE_BUFFER_ALGO )
  {
    // printf("Resetting n_spikes\n");
    *input_spike_buffer_ns::n_spikes_ = 0;
  }
  else
  {
    *SpikeNum = 0;
  }
}
