/*
 *  send_spike.h
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
 * @file send_spike.h
 * @brief Spike transmission system for neural communication
 *
 * This file defines the spike transmission system that handles sending
 * spikes from source neurons to their target neurons via synaptic
 * connections during simulation.
 *
 * Spike Transmission Architecture:
 * The system manages spike delivery through:
 * - Spike accumulation from firing neurons
 * - Connection-based spike routing
 * - Weight scaling (spike multiplication)
 * - Efficient batch spike processing
 *
 * Key Components:
 * - Spike queues: Temporary storage for spikes in current timestep
 * - Source tracking: Which neuron generated each spike
 * - Connection mapping: Which connections carry each spike
 * - Target counting: Number of targets for each spike
 * - Weight multiplication: Scaling factors for spike effects
 *
 * Data Structures:
 * - SpikeNum: Number of spikes generated per timestep
 * - SpikeSourceIdx: Source neuron for each spike
 * - SpikeConnIdx: Connection index for each spike
 * - SpikeMul: Weight multiplier for each spike
 * - SpikeTargetNum: Number of targets for spike multiplexing
 *
 * Transmission Flow:
 * 1. Neuron fires → generates spike
 * 2. Spike stored in queue with source, connection, multiplier
 * 3. Batch processing of all spikes in timestep
 * 4. Delivery to target neurons based on delays
 *
 * GPU Optimization:
 * - Coalesced memory writes for spike data
 * - Parallel spike generation from multiple neurons
 * - Efficient spike queue management
 * - Batch operations for spike delivery
 *
 * Performance Considerations:
 * - Memory bandwidth limited by spike data volume
 * - Queue size affects memory usage and cache efficiency
 * - Parallel generation from multiple neurons
 * - Efficient delivery to targets with different delays
 *
 * Spike Multiplication:
 * - Single spike can have multiple effects (multiplication)
 * - Used for synaptic efficacy scaling
 * - Supports different weight distributions
 *
 * @see spike_buffer.h Spike storage and temporal organization
 * @see get_spike.h Spike retrieval and delivery to targets
 * @see Connection Connection management system
 */

#ifndef SENDSPIKE_H
#define SENDSPIKE_H

/* Device Memory Arrays */
extern int* d_SpikeNum;         /**< Array of spike counts per timestep */
extern int* d_SpikeSourceIdx;   /**< Source neuron indices for each spike */
extern int* d_SpikeConnIdx;     /**< Connection indices for each spike */
extern float* d_SpikeMul;       /**< Weight multipliers for each spike */
extern int* d_SpikeTargetNum;   /**< Number of targets for spike multiplexing */

/* Device Constants and Pointers */
extern __device__ int MaxSpikeNum;       /**< Maximum number of spikes per timestep */
extern __device__ int* SpikeNum;         /**< Device pointer to spike counts */
extern __device__ int* SpikeSourceIdx;   /**< Device pointer to source indices */
extern __device__ int* SpikeConnIdx;     /**< Device pointer to connection indices */
extern __device__ float* SpikeMul;       /**< Device pointer to spike multipliers */
extern __device__ int* SpikeTargetNum;   /**< Device pointer to target counts */

/**
 * @brief GPU kernel to initialize spike data structures
 *
 * Initializes all spike transmission arrays to their default state,
 * preparing them for use during simulation.
 *
 * @param spike_num Array for spike counts
 * @param spike_source_idx Array for source neuron indices
 * @param spike_conn_idx Array for connection indices
 * @param spike_mul Array for spike multipliers
 * @param spike_target_num Array for target counts
 * @param max_spike_num Maximum number of spikes per timestep
 *
 * @note Called during network calibration
 */
__global__ void DeviceSpikeInit( int* spike_num,
  int* spike_source_idx,
  int* spike_conn_idx,
  float* spike_mul,
  int* spike_target_num,
  int max_spike_num );

/**
 * @brief Device function to send a spike
 *
 * Queues a spike for transmission from a source neuron through a
 * specific connection with a given weight multiplier.
 *
 * @param i_source Index of source neuron
 * @param i_conn Index of connection for this spike
 * @param mul Weight multiplier for spike effect
 * @param target_num Number of targets (for multiplexing)
 *
 * @note Thread-safe for concurrent spike generation
 * @warning May fail if spike queue is full
 *
 * Usage:
 * - Called by neuron update kernels when threshold is crossed
 * - Automatically stores spike for delayed delivery
 */
__device__ void SendSpike( int i_source, int i_conn, float mul, int target_num );

/**
 * @brief Initialize spike transmission system
 *
 * Allocates and initializes all spike transmission data structures.
 * Called during network calibration to prepare for simulation.
 *
 * @param max_spike_num Maximum number of spikes per timestep
 *
 * @note Must be called before simulation starts
 * @warning Allocates significant GPU memory
 */
void SpikeInit( int max_spike_num );

/**
 * @brief GPU kernel to reset spike data for new timestep
 *
 * Resets all spike transmission arrays for the beginning of a new
 * simulation timestep, clearing data from the previous timestep.
 *
 * @note Called at the beginning of each timestep
 */
__global__ void SpikeReset();

#endif
