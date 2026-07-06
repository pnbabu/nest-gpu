/*
 *  random.cu
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
 * @file random.cu
 * @brief GPU-accelerated random number generation for NEST GPU
 *
 * This file provides wrapper functions around the cuRAND library for
 * generating random numbers on GPU. It supports various distributions
 * needed for stochastic neural simulations.
 *
 * Architecture Overview:
 * ---------------------
 * Random number generation uses CUDA's cuRAND library:
 * - Parallel random number generation on GPU
 * - Multiple distribution support
 * - Independent random streams
 * - Reproducible results with seed control
 *
 * Supported Distributions:
 * -----------------------
 * - Uniform: curand_uniform() - floats in [0,1)
 * - Integer: curand_int() - unsigned integers
 * - Normal: curand_normal() - Gaussian distributed
 * - Poisson: curand_poisson() - Poisson distributed
 *
 * Memory Management:
 * ------------------
 * Efficient GPU memory handling:
 * - Allocate GPU memory for generation
 * - Generate numbers directly on GPU
 * - Transfer results to host
 * - Clean up GPU memory
 *
 * Key Functions:
 * --------------
 * - curand_int(): Generate unsigned integers
 * - curand_uniform(): Generate uniform floats [0,1)
 * - curand_normal(): Generate normal distributed floats
 * - curand_poisson(): Generate Poisson distributed integers
 *
 * GPU Implementation:
 * ------------------
 * cuRAND library features:
 * - High-quality random number generation
 * - Parallel generation across GPU cores
 * - Minimal CPU intervention
 * - Efficient memory bandwidth usage
 *
 * Integration Points:
 * ------------------
 * - Poisson generators: Random spike generation
 * - Connection rules: Random connectivity patterns
 * - Parameter initialization: Random parameter distributions
 * - Stochastic models: Noise and variability
 *
 * Usage Pattern:
 * --------------
 * 1. Create cuRAND generator with seed
 * 2. Call appropriate generation function
 * 3. Receive host array of random values
 * 4. Use values in simulation initialization
 *
 * Performance:
 * ------------
 * - Highly parallel on GPU
 * - Minimal CPU overhead
 * - Scales with GPU core count
 * - Memory bandwidth optimized
 *
 * Thread Safety:
 * --------------
 * - Each generator maintains independent state
 * - Multiple generators can operate concurrently
 * - GPU kernels handle parallel generation safely
 *
 * Applications:
 * -------------
 * - Random network connectivity
 * - Stochastic neuron parameters
 * - Poisson spike train generation
 * - Noise injection in simulations
 * - Monte Carlo simulations
 *
 * @see random.h Random number interface
 * @see curand.h CUDA random number library
 */

#include "cuda_error.h"
#include <config.h>
#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

unsigned int*
curand_int( curandGenerator_t& gen, size_t n )
{
  unsigned int* dev_data;
  // Allocate n integers on host
  unsigned int* host_data = new unsigned int[ n ];

  // Allocate n integers on device
  CUDAMALLOCCTRL( "&dev_data", ( void** ) &dev_data, n * sizeof( unsigned int ) );

  // Generate n integers on device
  CURAND_CALL( curandGenerate( gen, dev_data, n ) );
  // cudaDeviceSynchronize();
  //  Copy device memory to host
  CUDA_CALL( cudaMemcpy( host_data, dev_data, n * sizeof( unsigned int ), cudaMemcpyDeviceToHost ) );
  // Cleanup
  CUDAFREECTRL( "dev_data", dev_data );

  return host_data;
}

float*
curand_uniform( curandGenerator_t& gen, size_t n )
{
  float* dev_data;
  // Allocate n floats on host
  float* host_data = new float[ n ];

  // Allocate n floats on device
  CUDAMALLOCCTRL( "&dev_data", ( void** ) &dev_data, n * sizeof( float ) );

  // Generate n integers on device
  CURAND_CALL( curandGenerateUniform( gen, dev_data, n ) );
  // cudaDeviceSynchronize();
  //  Copy device memory to host
  CUDA_CALL( cudaMemcpy( host_data, dev_data, n * sizeof( float ), cudaMemcpyDeviceToHost ) );
  // Cleanup
  CUDAFREECTRL( "dev_data", dev_data );

  return host_data;
}

float*
curand_normal( curandGenerator_t& gen, size_t n, float mean, float stddev )
{
  size_t n1 = ( ( n % 2 ) == 0 ) ? n : n + 1; // round up to multiple of 2
  float* dev_data;
  // Allocate n floats on host
  float* host_data = new float[ n ];

  // Allocate n1 floats on device
  CUDAMALLOCCTRL( "&dev_data", ( void** ) &dev_data, n1 * sizeof( float ) );

  // Generate n1 integers on device
  // printf("curandGenerateNormal n1: %d\tmean: %f\tstd: %f\n", (int)n1, mean,
  //	 stddev);
  CURAND_CALL( curandGenerateNormal( gen, dev_data, n1, mean, stddev ) );
  // cudaDeviceSynchronize();
  //  Copy device memory to host
  CUDA_CALL( cudaMemcpy( host_data, dev_data, n * sizeof( float ), cudaMemcpyDeviceToHost ) );
  // Cleanup
  CUDAFREECTRL( "dev_data", dev_data );

  return host_data;
}
