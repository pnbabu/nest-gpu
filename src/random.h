/**
 * @file random.h
 * @brief Random number generation utilities for NEST GPU
 *
 * This file provides interface functions for GPU-accelerated random number
 * generation using the NVIDIA cuRAND library. Random numbers are essential
 * for various stochastic processes in neural simulations.
 *
 * Random Number Applications:
 * - Poisson spike generation: Random spike times
 * - Connection patterns: Random network connectivity
 * - Parameter randomization: Heterogeneous neuron properties
 * - Synaptic plasticity: Stochastic learning rules
 * - Noise injection: Background activity simulation
 *
 * GPU Random Number Generation:
 * - Uses NVIDIA cuRAND library for parallel generation
 * - Supports multiple distributions (uniform, normal, Poisson)
 * - High-throughput generation for large-scale simulations
 * - Reproducible sequences with fixed seeds
 *
 * Quality Considerations:
 * - High-quality pseudo-random numbers
 * - Suitable for scientific simulations
 * - Long period length for large-scale use
 * - Good statistical properties
 *
 * Performance:
 * - Parallel generation across GPU threads
 * - Memory bandwidth limited for large arrays
 * - Efficient for batch operations
 * - Optimized for neural simulation workloads
 *
 * Thread Safety:
 * - Thread-safe for concurrent access
 * - Each thread can access different random streams
 * - Sequential consistency guarantees
 *
 * Usage Patterns:
 * - Generate arrays of random numbers
 * - Batch operations for efficiency
 * - Device memory allocation and management
 * - Host-device memory transfer
 *
 * Dependencies:
 * - NVIDIA cuRAND library
 * - CUDA runtime
 * - GPU device with compute capability
 *
 * @see curand_kernel.h Device-side random number generation
 * @see poiss_gen.h Poisson-specific random generation
 */

#ifndef RANDOM_H
#define RANDOM_H
#include <curand.h>

/**
 * @brief Generate random integers using cuRAND
 *
 * Generates an array of random unsigned integers using the cuRAND
 * generator. Useful for indexing and discrete random variables.
 *
 * @param gen cuRAND generator handle (reference)
 * @param n Number of random integers to generate
 * @return Pointer to array of random integers (device memory)
 *
 * Properties:
 * - Range: 0 to UINT_MAX
 * - Distribution: Uniform over all 32-bit integers
 * - Quality: Suitable for scientific simulations
 * - Performance: ~1 GB/s on modern GPUs
 *
 * Usage:
 * @code
 * curandGenerator_t gen;
 * curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
 * curandSetPseudoRandomGeneratorSeed(gen, seed);
 *
 * unsigned int* random_ints = curand_int(gen, 1000);
 * // Use random_ints...
 * cudaFree(random_ints);
 * @endcode
 *
 * Memory Management:
 * - Allocates device memory for the output array
 * - Caller responsible for freeing the memory
 * - Returns NULL if generation fails
 *
 * @note Thread-safe for concurrent calls
 * @warning Caller must free returned pointer
 */
unsigned int* curand_int( curandGenerator_t& gen, size_t n );

/**
 * @brief Generate random floats with uniform distribution
 *
 * Generates an array of random floating-point numbers uniformly
 * distributed in the range [0.0, 1.0).
 *
 * @param gen cuRAND generator handle (reference)
 * @param n Number of random floats to generate
 * @return Pointer to array of random floats (device memory)
 *
 * Properties:
 * - Range: [0.0, 1.0) (inclusive of 0.0, exclusive of 1.0)
 * - Distribution: Uniform
 * - Precision: 32-bit floating point
 * - Quality: High-quality random numbers
 *
 * Usage:
 * @code
 * curandGenerator_t gen;
 * curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
 * curandSetPseudoRandomGeneratorSeed(gen, seed);
 *
 * float* random_floats = curand_uniform(gen, 10000);
 * // Transform to desired range: a + (b-a) * random_floats[i]
 * cudaFree(random_floats);
 * @endcode
 *
 * Applications:
 * - Probability sampling
 * - Uniform random connections
 * - Random parameter initialization
 * - Monte Carlo simulations
 *
 * @note Thread-safe for concurrent calls
 * @warning Caller must free returned pointer
 */
float* curand_uniform( curandGenerator_t& gen, size_t n );

/**
 * @brief Generate random floats with normal (Gaussian) distribution
 *
 * Generates an array of random floating-point numbers following a
 * normal (Gaussian) distribution with specified mean and standard deviation.
 *
 * @param gen cuRAND generator handle (reference)
 * @param n Number of random floats to generate
 * @param mean Mean of the distribution (μ)
 * @param stddev Standard deviation of the distribution (σ)
 * @return Pointer to array of random floats (device memory)
 *
 * Properties:
 * - Distribution: Normal N(mean, stddev²)
 * - PDF: f(x) = (1/(σ√(2π))) * exp(-((x-μ)²/(2σ²)))
 * - Precision: 32-bit floating point
 * - Quality: Box-Muller transform or ziggurat algorithm
 *
 * Mathematical Background:
 * The normal distribution is defined by:
 * @f[
 * f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
 * @f]
 *
 * Usage:
 * @code
 * curandGenerator_t gen;
 * curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
 * curandSetPseudoRandomGeneratorSeed(gen, seed);
 *
 * // Generate 1000 random values: mean=0.0, std=1.0
 * float* gaussian_noise = curand_normal(gen, 1000, 0.0f, 1.0f);
 * // Add to neuron membrane potentials
 * cudaFree(gaussian_noise);
 * @endcode
 *
 * Applications:
 * - Gaussian noise injection
 * - Parameter randomization (normal distribution)
 * - Synaptic weight initialization
 * - Background activity modeling
 *
 * Performance:
 * - Slightly slower than uniform distribution
 * - Uses transformation algorithm
 * - Memory bandwidth limited
 *
 * @note Thread-safe for concurrent calls
 * @warning Caller must free returned pointer
 * @see curand_normal_clipped for bounded normal distribution
 */
float* curand_normal( curandGenerator_t& gen, size_t n, float mean, float stddev );

#endif
