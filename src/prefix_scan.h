/*
 *  prefix_scan.h
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
 * @file prefix_scan.h
 * @brief Parallel prefix scan (cumulative sum) operations
 *
 * This file defines the PrefixScan class for efficient parallel prefix scan
 * operations on GPU. Prefix scans are fundamental parallel primitives used
 * throughout NEST GPU for various computational tasks.
 *
 * Mathematical Definition:
 * Given an input array [a₀, a₁, a₂, ..., a_{n-1}], the inclusive prefix scan
 * produces [a₀, a₀+a₁, a₀+a₁+a₂, ..., a₀+a₁+...+a_{n-1}].
 *
 * The exclusive prefix scan produces [0, a₀, a₀+a₁, ..., a₀+...+a_{n-2}].
 *
 * Algorithm Overview:
 * The implementation uses work-efficient parallel scan algorithms:
 * - Up-sweep (reduce) phase: Build partial sums in tree structure
 * - Down-sweep phase: Distribute partial sums to final positions
 * - O(n log n) work complexity, O(log n) time complexity
 *
 * GPU Implementation:
 * - Optimized for CUDA architecture
 * - Uses shared memory for efficiency
 * - Handles arbitrary array sizes
 * - Bank conflict avoidance
 *
 * Applications in NEST GPU:
 * - Spike count accumulation per neuron
 * - Connection index calculation
 * - Array indexing for sparse data structures
 * - Parallel prefix operations in connectivity
 *
 * Performance Characteristics:
 * - Memory bandwidth limited for large arrays
 * - Efficient use of shared memory
 * - Optimized memory access patterns
 * - O(log n) parallel steps
 *
 * Memory Requirements:
 * - Requires temporary storage for intermediate results
 * - Typically 2-3x the input array size
 * - Device memory allocation and management
 *
 * Usage Pattern:
 * @code
 * PrefixScan scanner;
 * scanner.Init();  // Allocate resources
 *
 * // Perform scan on array of size n
 * scanner.Scan(d_output, d_input, n);
 *
 * scanner.Free();  // Release resources
 * @endcode
 *
 * @see scan.h Alternative scan implementations
 * @see nested_loop.h Scan usage in connectivity algorithms
 */

#ifndef PREFIXSCAN_H
#define PREFIXSCAN_H

/**
 * @class PrefixScan
 * @brief Parallel prefix scan implementation for GPU
 *
 * This class provides efficient parallel prefix scan operations using
 * work-efficient algorithms optimized for CUDA architecture.
 *
 * Algorithm Choice:
 * - Work-efficient: O(n log n) total work
 * - Step-efficient: O(log n) parallel steps
 * - Memory-efficient: Minimal temporary storage
 *
 * GPU Optimization:
 * - Shared memory usage for fast access
 * - Coalesced global memory reads/writes
 * - Bank conflict avoidance
 * - Efficient warp utilization
 *
 * Supported Operations:
 * - Inclusive scan: output[i] = sum(input[0..i])
 * - Exclusive scan: output[i] = sum(input[0..i-1])
 * - Various data types (int, float, etc.)
 *
 * Resource Management:
 * - Init(): Allocate GPU memory and resources
 * - Scan(): Perform prefix scan operation
 * - Free(): Release allocated resources
 *
 * Error Handling:
 * - Returns status codes for operations
 * - Validates input parameters
 * - Handles memory allocation failures
 *
 * Usage Considerations:
 * - Call Init() before first scan
 * - Call Free() when done
 * - Can perform multiple scans between Init() and Free()
 * - Thread-safe for multiple instances
 */
class PrefixScan
{
public:
  static const unsigned int AllocSize; /**< Maximum allocation size for temporary storage */

  /* Internal Memory Pointers (commented out but present for reference)
  uint *d_Input;    /**< Device input array pointer */
  uint *d_Output;   /**< Device output array pointer */
  uint *h_Input;    /**< Host input array pointer */
  uint *h_OutputCPU; /**< Host CPU output array pointer */
  uint *h_OutputGPU; /**< Host GPU output array pointer */
  */

  /**
   * @brief Initialize prefix scan resources
   *
   * Allocates GPU memory and initializes resources needed for
   * prefix scan operations. Must be called before first Scan().
   *
   * @return 0 on success, error code on failure
   *
   * Resource Allocation:
   * - Temporary device memory for intermediate results
   * - CUDA events for timing (if enabled)
   * - Stream management (if needed)
   *
   * @note Must be called before first Scan() operation
   * @warning Allocates significant GPU memory
   * @see Free() for cleanup
   */
  int Init();

  /**
   * @brief Perform parallel prefix scan
   *
   * Computes the inclusive prefix scan of the input array and stores
   * the result in the output array.
   *
   * @param d_Output Output array (device memory)
   * @param d_Input Input array (device memory)
   * @param n Number of elements to scan
   * @return 0 on success, error code on failure
   *
   * Algorithm:
   * 1. Up-sweep phase: Compute partial sums in tree
   * 2. Down-sweep phase: Distribute to final positions
   * 3. Handle arbitrary array sizes with padding
   *
   * Performance:
   * - O(log n) parallel steps
   * - Memory bandwidth limited
   * - Efficient for large arrays (>1000 elements)
   *
   * Memory Access:
   * - Reads from d_Input
   * - Writes to d_Output
   * - Uses temporary storage during computation
   *
   * @note Both arrays must be in device memory
   * @note Input and output can be the same array (in-place)
   * @warning Both pointers must be valid device memory
   */
  int Scan( int* d_Output, int* d_Input, int n );

  /**
   * @brief Release prefix scan resources
   *
   * Frees all GPU memory and resources allocated by Init().
   * Should be called when prefix scan operations are complete.
   *
   * @return 0 on success, error code on failure
   *
   * Cleanup Operations:
   * - Free device memory allocations
   * - Destroy CUDA events/streams
   * - Reset internal state
   *
   * @note Should be called when done with prefix scan operations
   * @see Init() for resource allocation
   */
  int Free();
};

#endif
