/*
 *  spike_buffer.h
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
 * @file spike_buffer.h
 * @brief Spike buffer management for neural simulation
 *
 * This file defines the spike buffer system that manages the storage,
 * organization, and delivery of spikes between neurons during simulation.
 *
 * Spike Buffer Architecture:
 * The spike buffer system provides efficient spike communication through:
 * - Ring buffers for spike storage with temporal organization
 * - Per-neuron spike buffers with configurable size limits
 * - Efficient spike delivery based on connection delays
 * - Support for spike multiplication (weight scaling)
 *
 * Key Components:
 * - Ring buffers: Circular buffers for spike storage
 * - Delay lines: Spike delivery based on synaptic delays
 * - Buffer indexing: Efficient lookup of spikes per timestep
 * - Spike multiplication: Support for spike weight scaling
 *
 * Data Structures:
 * - LastSpikeMul: Multiplier for last spike from each buffer
 * - LastSpikeTimeIdx: Time index of last spike from each buffer
 * - ConnectionSpikeTime: Next spike time for each connection
 * - SpikeBufferSize: Number of spikes in each buffer
 * - SpikeBufferIdx0: Index of most recent spike in buffer
 * - SpikeBufferTimeIdx: Time indices of buffered spikes
 * - SpikeBufferConnIdx: Connection indices for buffered spikes
 * - SpikeBufferMul: Spike multipliers for buffered spikes
 *
 * Buffer Organization:
 * - Ring buffer structure for efficient memory usage
 * - Circular indexing to avoid buffer overflow
 * - Temporal organization for delay-based delivery
 * - Per-neuron buffers for independent spike handling
 *
 * GPU Optimization:
 * - Coalesced memory access patterns
 * - Shared memory for frequently accessed buffers
 * - Efficient ring buffer indexing
 * - Parallel spike delivery
 *
 * Performance Considerations:
 * - Memory bandwidth is often the bottleneck
 * - Buffer size affects memory usage and cache efficiency
 * - Ring buffer overflow protection
 * - Efficient spike retrieval for target neurons
 *
 * MPI Support:
 * - External spike flag for cross-host spikes
 * - Remote spike buffer management
 * - Distributed spike communication
 *
 * @see Connection Connection management system
 * @see send_spike.h Spike transmission between neurons
 * @see get_spike.h Spike retrieval and delivery
 */

#ifndef SPIKEBUFFER_H
#define SPIKEBUFFER_H

/* GPU Device Constants */
extern __constant__ bool ExternalSpikeFlag; /**< Enable external spike handling */
extern __device__ int MaxSpikeBufferSize;   /**< Maximum size of spike buffers */
extern __device__ int NSpikeBuffer;         /**< Number of spike buffers */

/* Host Variables */
extern int h_NSpikeBuffer; /**< Host-side number of spike buffers */

/* Spike Data Arrays (Device Memory) */
extern float* d_LastSpikeMul;          /**< Array of last spike multipliers [NSpikeBuffer] */
extern __device__ float* LastSpikeMul; /**< Device pointer to last spike multipliers */

extern long long* d_LastSpikeTimeIdx;          /**< Array of last spike time indices [NSpikeBuffer] */
extern __device__ long long* LastSpikeTimeIdx; /**< Device pointer to last spike time indices */

extern long long* d_LastRevSpikeTimeIdx;          /**< Array of last reverse spike time indices [NSpikeBuffer] */
extern __device__ long long* LastRevSpikeTimeIdx; /**< Device pointer to last reverse spike time indices */

extern unsigned short* d_ConnectionSpikeTime;          /**< Next spike time for each connection [NConnection] */
extern __device__ unsigned short* ConnectionSpikeTime; /**< Device pointer to connection spike times */

/* Spike Buffer Management Arrays */
extern int* d_SpikeBufferSize;    /**< Number of spikes in each buffer */
extern __device__ int* SpikeBufferSize; /**< Device pointer to spike buffer sizes */

extern int* d_SpikeBufferIdx0;    /**< Index of most recent spike in each buffer */
extern __device__ int* SpikeBufferIdx0; /**< Device pointer to buffer indices */

extern int* d_SpikeBufferTimeIdx; /**< Time indices of buffered spikes */
extern __device__ int* SpikeBufferTimeIdx; /**< Device pointer to spike time indices */

extern int* d_SpikeBufferConnIdx; /**< Connection indices for buffered spikes */
extern __device__ int* SpikeBufferConnIdx; /**< Device pointer to connection indices */

extern float* d_SpikeBufferMul;    /**< Spike multipliers for buffered spikes */
extern __device__ float* SpikeBufferMul; /**< Device pointer to spike multipliers */

/**
 * @brief Device function to push a spike to a buffer
 *
 * Pushes a spike with specified multiplier to the designated spike buffer.
 * Uses ring buffer indexing for efficient circular storage.
 *
 * @param i_spike_buffer Index of the spike buffer
 * @param mul Spike multiplier (weight scaling factor)
 *
 * @note Thread-safe for concurrent access
 * @warning May fail if buffer is full
 */
__device__ void PushSpike( int i_spike_buffer, float mul );

/**
 * @brief GPU kernel to update spike buffer state
 *
 * Updates the spike buffers for the current timestep, including:
 * - Advancing ring buffer indices
 * - Updating spike time indices
 * - Managing buffer overflow
 *
 * @note Called once per simulation timestep
 */
__global__ void SpikeBufferUpdate();

/**
 * @brief GPU kernel to initialize spike buffers on device
 *
 * Initializes all spike buffer data structures to their default state,
 * preparing them for use during simulation.
 *
 * @param n_spike_buffers Number of spike buffers to initialize
 * @param max_spike_buffer_size Maximum size of each buffer
 * @param last_spike_time_idx Array for last spike time indices
 * @param last_spike_mul Array for last spike multipliers
 * @param spike_buffer_size Array for buffer sizes
 * @param spike_buffer_idx0 Array for buffer indices
 * @param spike_buffer_time Array for spike time indices
 * @param spike_buffer_conn Array for connection indices
 * @param spike_buffer_mul Array for spike multipliers
 * @param last_rev_spike_time_idx Array for reverse spike time indices
 *
 * @note Called during network calibration
 */
__global__ void DeviceSpikeBufferInit( int n_spike_buffers,
  int max_spike_buffer_size,
  long long* last_spike_time_idx,
  float* last_spike_mul,
  int* spike_buffer_size,
  int* spike_buffer_idx0,
  int* spike_buffer_time,
  int* spike_buffer_conn,
  float* spike_buffer_mul,
  long long* last_rev_spike_time_idx );

/**
 * @brief Initialize spike buffer system
 *
 * Allocates and initializes all spike buffer data structures for the
 * network. This is called during network calibration to prepare for
 * simulation.
 *
 * @param n_local_nodes Number of local neurons
 * @param n_image_nodes Number of image neurons (for remote connections)
 * @param max_spike_buffer_size Maximum size of each spike buffer
 * @param spike_buffer_algo Algorithm for spike buffer management
 *
 * @return 0 on success, error code on failure
 *
 * @note Must be called before simulation starts
 * @warning Allocates significant GPU memory
 */
int spikeBufferInit( uint n_local_nodes, uint n_image_nodes, int max_spike_buffer_size, int spike_buffer_algo );

#endif
