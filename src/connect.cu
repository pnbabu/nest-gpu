/*
 *  connect.cu
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
 * @file connect.cu
 * @brief GPU-accelerated connectivity management for spiking neural networks
 *
 * This file implements the core connectivity infrastructure for NEST GPU,
 * handling creation, storage, and management of synaptic connections between
 * neurons. It provides efficient GPU-based algorithms for network construction
 * and spike delivery.
 *
 * Architecture Overview:
 * ---------------------
 * The connectivity system uses a compressed sparse representation optimized
 * for GPU processing. Connections are organized by source neurons to enable
 * efficient spike delivery during simulation.
 *
 * Connection Storage:
 * ------------------
 * Connections are stored using a grouped sparse format:
 * - ConnGroupIdx0: Index of first connection group for each source neuron
 * - ConnGroupIConn0: Index of first connection in each group
 * - ConnGroupDelay: Delay associated with each connection group
 * - ConnGroupDelayRate: Parameter for delay-based routing
 * - ConnGroupStart: First connection index in each group
 * - ConnGroupEnd: Last connection index + 1 in each group
 *
 * Bit-Packed Connection Data:
 * ---------------------------
 * Individual connections are stored in compressed format using bit packing:
 * - Source neuron ID: MaxNodeNBits bits
 * - Delay steps: MaxDelayNBits bits
 * - Target neuron ID: Remaining bits
 * - Synapse group and port info: Additional fields
 *
 * This compression reduces memory footprint and improves cache efficiency.
 *
 * Connection Rules:
 * ----------------
 * Supported connectivity patterns:
 * - all_to_all: Every source connects to every target
 * - one_to_one: Paired connections between source and target
 * - fixed_indegree: Each target receives fixed number of random inputs
 * - fixed_outdegree: Each source sends to fixed number of random targets
 * - fixed_total_number: Fixed number of random connections
 *
 * GPU Implementation:
 * ------------------
 * Connection creation uses parallel algorithms:
 * - curand for GPU-based random number generation
 * - CUB library for efficient parallel primitives
 * - Bit manipulation for connection data packing
 * - Coalesced memory access patterns
 *
 * Key Algorithms:
 * --------------
 * 1. Connection Generation:
 *    - Parallel random connection creation
 *    - Duplicate detection and removal
 *    - Weight and delay assignment
 *
 * 2. Connection Sorting:
 *    - Sort by source neuron for efficient spike delivery
 *    - Group connections by (source, delay, synapse_type)
 *    - Build index structures for fast access
 *
 * 3. Spike Delivery:
 *    - Use grouped structure to minimize branching
 *    - Efficient scattering of synaptic inputs
 *    - Delay-based routing through spike buffers
 *
 * Memory Management:
 * -----------------
 * - Dynamic allocation based on network size
 * - Pinned memory for CPU-GPU transfers
 * - Efficient memory alignment for coalesced access
 * - Compression to reduce memory requirements
 *
 * Bit Packing Details:
 * ------------------
 * The system uses bit-level compression to store connection data:
 * - 32-bit integers store multiple connection attributes
 * - Bit masks extract individual components
 * - Configurable bit allocation for different attributes
 * - Trade-off between compression ratio and decompression speed
 *
 * Connection Grouping:
 * --------------------
 * Connections with same (source, delay, synapse_type) are grouped:
 * - Reduces duplicate delay information storage
 * - Enables bulk spike delivery operations
 * - Improves memory access patterns
 * - Simplifies spike buffer management
 *
 * Integration Points:
 * ------------------
 * - NESTGPU: Main simulation engine uses connectivity for spike delivery
 * - Spike buffers: Use connection delays for routing
 * - Neuron models: Receive synaptic inputs through connections
 * - MPI: Distributed connection management
 *
 * Performance Considerations:
 * ---------------------------
 * - Memory bandwidth is primary bottleneck
 * - Coalesced access patterns are critical
 * - Minimize thread divergence in connection kernels
 * - Use shared memory for frequently accessed connection data
 * - Batch connection creation for efficiency
 *
 * Thread Safety:
 * --------------
 * - Connection creation is not thread-safe
 * - Spike delivery using connections is thread-safe on GPU
 * - Multiple networks require separate connection instances
 *
 * Usage Pattern:
 * --------------
 * 1. Define connection rule and parameters
 * 2. Call Connect() with source and target neuron groups
 * 3. System creates connections using GPU-accelerated algorithms
 * 4. Connections are sorted and indexed for efficient access
 * 5. During simulation, use connections for spike delivery
 *
 * Mathematical Foundation:
 * -----------------------
 * Connection probabilities and random selection:
 * - Fixed indegree: Hypergeometric distribution
 * - Fixed outdegree: Binomial distribution
 * - Uniform random selection with GPU-accelerated sampling
 *
 * @see connect.h Connectivity interface and classes
 * @see connect_rules.h Connection rule implementations
 * @see nestgpu.h Main simulation engine
 */

// #include <time.h>
// #include <sys/time.h>
#include "cuda_error.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda.h>
#include <curand.h>
#include <iostream>
#include <utility>
#include <vector>
// #include "copass_kernels.h"
// #include "copass_sort.h"
// #include "distribution.h"
#include "connect.h"
#include "nestgpu.h"
// #include "utilities.h"

// #define OPTIMIZE_FOR_MEMORY

// extern __constant__ float NESTGPUTimeResolution;

bool print_sort_err = true;
bool print_sort_cfr = false;
bool compare_with_serial = false;
uint last_i_sub = 0;

// maximum number of bits used to represent node index
__device__ int MaxNodeNBits;

// maximum number of bits used to represent delays
__device__ int MaxDelayNBits;

// maximum number of bits used to represent synapse group index
__device__ int MaxSynNBits;

// maximum number of bits used to represent receptor port index
__device__ int MaxPortNBits;

// maximum number of bits used to represent receptor port index
// and synapse group index
__device__ int MaxPortSynNBits;

// bit mask used to extract source node index
__device__ uint SourceMask;

// bit mask used to extract delay
__device__ uint DelayMask;

// bit mask used to extract target node index
__device__ uint TargetMask;

// bit mask used to extract synapse group index
__device__ uint SynMask;

// bit mask used to extract port index
__device__ uint PortMask;

// bit mask used to extract port and synapse group index
__device__ uint PortSynMask;

// minimum allowed delay in time step units
__constant__ uint MinAllowedDelay;

// ig0 = ConnGroupIdx0[i_spike_buffer] is the index in the whole
// connection-group array of the first connection group outgoing
// from the node i_spike_buffer
__device__ iconngroup_t* ConnGroupIdx0;

// i_conn0 = ConnGroupIConn0[ig] with ig = 0, ..., Ng
//  is the index in the whole connection array of the first connection
// belonging to the connection group ig
__device__ int64_t* ConnGroupIConn0;

// ConnGroupDelay[ig]
// delay associated to all connections of the connection group ig
// with ig = 0, ..., Ng
__device__ int* ConnGroupDelay;

// size (i.e. number of connections) of connection blocks
// int64_t h_ConnBlockSize = 10000000; // 160000000; //50000000;
__device__ int64_t ConnBlockSize;

// Array of source node indexes and delays of all connections
// Source node indexes and delays are merged in a single integer variable
// The most significant MaxNodeNBits are used for the node index
// the others (less significant) bits are used to represent the delay
// This array is used as a key array for sorting the connections
// in ascending order according to the source node index
// Connections from the same source node are sorted according to
// the delay
// it seems that there is no relevant advantage in using a constant array
// however better to keep this option ready and commented
//__constant__ uint* ConnKeyArray[1024];
__device__ void* ConnKeyArray;

// array of target node indexes, receptor port index, synapse type,
// weight of all connections
// used as a value for key-value sorting of the connections (see above)
// it seems that there is no relevant advantage in using a constant array
// however better to keep this option ready and commented
//__constant__ connection_struct* ConnStructArray[1024];
__device__ void* ConnStructArray;

__device__ unsigned short* ConnectionSpikeTime;

const std::string ConnectionFloatParamName[ N_CONN_FLOAT_PARAM ] = { "weight", "delay" };

const std::string ConnectionIntParamName[ N_CONN_INT_PARAM ] = { "source", "target", "port", "syn_group" };

__global__ void
setConnGroupNum( inode_t n_compact,
  iconngroup_t* conn_group_num,
  iconngroup_t* conn_group_idx0_compact,
  inode_t* conn_group_source_compact )
{
  inode_t i_compact = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_compact >= n_compact )
  {
    return;
  }
  inode_t source = conn_group_source_compact[ i_compact ];
  iconngroup_t num = conn_group_idx0_compact[ i_compact + 1 ] - conn_group_idx0_compact[ i_compact ];
  conn_group_num[ source ] = num;
}

__global__ void
setConnGroupIConn0( int64_t n_block_conn,
  int* conn_group_iconn0_mask,
  iconngroup_t* conn_group_iconn0_mask_cumul,
  int64_t* conn_group_iconn0,
  int64_t i_conn0,
  iconngroup_t* offset )
{
  int64_t i_conn = threadIdx.x + blockIdx.x * blockDim.x;
  if ( i_conn >= n_block_conn )
  {
    return;
  }
  if ( conn_group_iconn0_mask[ i_conn ] != 0 )
  {
    iconngroup_t pos = conn_group_iconn0_mask_cumul[ i_conn ] + *offset;
    conn_group_iconn0[ pos ] = i_conn0 + i_conn;
  }
}

__global__ void
connectCalibrateKernel( iconngroup_t* conn_group_idx0,
  int64_t* conn_group_iconn0,
  int* conn_group_delay,
  int64_t block_size,
  void* conn_key_array,
  void* conn_struct_array,
  unsigned short* conn_spike_time )
{
  ConnGroupIdx0 = conn_group_idx0;
  ConnGroupIConn0 = conn_group_iconn0;
  ConnGroupDelay = conn_group_delay;
  ConnBlockSize = block_size;
  ConnKeyArray = conn_key_array;
  ConnStructArray = conn_struct_array;
  ConnectionSpikeTime = conn_spike_time;
}

__global__ void
setSourceTargetIndexKernel( uint64_t n_src_tgt,
  uint n_source,
  uint n_target,
  uint64_t* d_src_tgt_arr,
  uint* d_src_arr,
  uint* d_tgt_arr )
{
  uint64_t i_src_tgt = ( uint64_t ) blockIdx.x * blockDim.x + threadIdx.x;
  if ( i_src_tgt >= n_src_tgt )
  {
    return;
  }
  uint i_src = ( uint ) ( i_src_tgt / n_target );
  uint i_tgt = ( uint ) ( i_src_tgt % n_target );
  uint src_id = d_src_arr[ i_src ];
  uint tgt_id = d_tgt_arr[ i_tgt ];
  uint64_t src_tgt_id = ( ( uint64_t ) src_id << 32 ) | tgt_id;
  d_src_tgt_arr[ i_src_tgt ] = src_tgt_id;
  // printf("i_src_tgt %lld\tsrc_id %d\ttgt_id %d\tsrc_tgt_id %lld\n",
  //	 i_src_tgt, src_id, tgt_id, src_tgt_id);
}

// Get the index of the connection float parameter param_name
// if param_name is not a float parameter, return -1
int
Connection::getConnectionFloatParamIndex( std::string param_name )
{
  for ( int i = 0; i < N_CONN_FLOAT_PARAM; i++ )
  {
    if ( param_name == ConnectionFloatParamName[ i ] )
    {
      return i;
    }
  }

  return -1;
}

// Get the index of the connection int parameter param_name
// if param_name is not an int parameter, return -1
int
Connection::getConnectionIntParamIndex( std::string param_name )
{
  for ( int i = 0; i < N_CONN_INT_PARAM; i++ )
  {
    if ( param_name == ConnectionIntParamName[ i ] )
    {
      return i;
    }
  }

  return -1;
}

// Check if param_name is a connection float parameter
int
Connection::isConnectionFloatParam( std::string param_name )
{
  if ( getConnectionFloatParamIndex( param_name ) >= 0 )
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

// Check if param_name is a connection integer parameter
int
Connection::isConnectionIntParam( std::string param_name )
{
  if ( getConnectionIntParamIndex( param_name ) >= 0 )
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

// set minimum allowed delay
int
Connection::setMinAllowedDelay(float min_allowed_delay_float)
{
  int min_allowed_delay = (int) round (min_allowed_delay_float / time_resolution_);
  if (min_allowed_delay < 1) {
    throw ngpu_exception("Minimum allowed delay cannot be smaller than the time resolution");
  }
  min_allowed_delay_ = min_allowed_delay; // note that min_allowed_delay_ is unsigned
                                          // this is why an additional (signed) variable is used
                                          // for the previous check

  // std::cout << "min_allowed_delay_: " << min_allowed_delay_ << "\n";
  gpuErrchk( cudaMemcpyToSymbol( MinAllowedDelay, &min_allowed_delay_, sizeof( uint ) ) );

  return 0;
}

// get minimum allowed delay
float
Connection::getMinAllowedDelay()
{
  return time_resolution_ * min_allowed_delay_;
}
