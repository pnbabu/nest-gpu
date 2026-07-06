/*
 *  nestgpu.h
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
 * @file nestgpu.h
 * @brief Main NEST GPU engine interface class definition
 *
 * This file defines the core NESTGPU class that manages GPU-accelerated
 * spiking neural network simulations. It provides a Python-like interface
 * compatible with the NEST simulator but optimized for GPU execution.
 *
 * Key Functionality:
 * - Network creation and neuron model management
 * - Connection setup with various connectivity rules
 * - Simulation control and time management
 * - Multi-GPU and MPI support for distributed simulations
 * - Parameter access and modification
 * - Data recording and retrieval
 *
 * Architecture:
 * - Manages node groups for efficient GPU memory organization
 * - Handles spike communication between neurons
 * - Coordinates host-device memory transfers
 * - Supports multiple connection storage structures
 * - Provides random number generation utilities
 *
 * Usage Pattern:
 * 1. Create NESTGPU instance
 * 2. Set simulation parameters (time resolution, random seed)
 * 3. Create neurons with Create()
 * 4. Connect neurons with Connect()
 * 5. Calibrate the network
 * 6. Run simulation with Simulate()
 * 7. Retrieve recorded data
 *
 * Performance Considerations:
 * - Optimized for large-scale networks (>1000 neurons)
 * - Efficient spike communication patterns
 * - Memory coalescing for GPU kernels
 * - Supports multiple GPU configurations
 *
 * @see BaseNeuron Base class for all neuron models
 * @see Connection Connection management system
 * @see SynSpec Synapse specifications
 */

#ifndef NESTGPU_H
#define NESTGPU_H

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "base_neuron.h"
#include "connect_spec.h"
#include "ngpu_exception.h"
#include "node_group.h"
// #include "connect.h"
// #include "syn_model.h"
// #include "distribution.h"

class Multimeter;

struct curandGenerator_st;

typedef struct curandGenerator_st* curandGenerator_t;

class ConnSpec;

class SynSpec;

class SynModel;

class Connection;

typedef uint inode_t;   /**< Integer type for node indices */
typedef uint iconngroup_t; /**< Integer type for connection group indices */

/**
 * @class Sequence
 * @brief Represents a contiguous sequence of integers
 *
 * This class is used extensively throughout NEST GPU to represent
 * sequences of node indices, connection indices, etc. It provides
 * efficient indexing and slicing operations without storing the
 * full sequence in memory.
 *
 * Memory Efficiency:
 * - Only stores start index (i0) and length (n)
 * - Represents sequence [i0, i0+1, ..., i0+n-1]
 * - Constant memory regardless of sequence length
 *
 * Usage Examples:
 * @code
 * Sequence seq(10, 5);  // Represents [10, 11, 12, 13, 14]
 * int val = seq[2];     // Returns 12
 * Sequence sub = seq.Subseq(1, 3); // Represents [11, 12, 13]
 * std::vector<int> vec = seq.ToVector(); // Converts to [10,11,12,13,14]
 * @endcode
 *
 * Error Handling:
 * - Throws ngpu_exception for out-of-bounds access
 * - Validates index ranges in all operations
 *
 * @see NodeSeq Specialized sequence for node indices
 * @see RemoteNodeSeq Sequence with host information
 */
class Sequence
{
public:
  int i0; /**< Starting index of the sequence */
  int n;  /**< Length of the sequence */

  /**
   * @brief Constructor for Sequence
   * @param i0 Starting index (default: 0)
   * @param n Length of sequence (default: 0)
   */
  Sequence( int i0 = 0, int n = 0 )
    : i0( i0 )
    , n( n )
  {
  }

  /**
   * @brief Index operator with bounds checking
   * @param i Index within sequence (0 <= i < n)
   * @return Actual value at position i (i0 + i)
   * @throws ngpu_exception if index is out of bounds
   */
  inline int
  operator[]( int i )
  {
    if ( i < 0 )
    {
      throw ngpu_exception( "Sequence index cannot be negative" );
    }
    if ( i >= n )
    {
      throw ngpu_exception( "Sequence index out of range" );
    }
    return i0 + i;
  }

  /**
   * @brief Create a subsequence from this sequence
   * @param first First index in subsequence (inclusive)
   * @param last Last index in subsequence (inclusive)
   * @return New Sequence representing the subsequence
   * @throws ngpu_exception if range is invalid
   *
   * Example: seq.Subseq(2, 5) creates sequence containing
   * elements at indices 2, 3, 4, 5 from the original sequence.
   */
  inline Sequence
  Subseq( int first, int last )
  {
    if ( first < 0 || first > last )
    {
      throw ngpu_exception( "Sequence subset range error" );
    }
    if ( last >= n )
    {
      throw ngpu_exception( "Sequence subset out of range" );
    }
    return Sequence( i0 + first, last - first + 1 );
  }

  /**
   * @brief Convert sequence to std::vector<int>
   * @return Vector containing all elements in the sequence
   *
   * This method materializes the sequence into an actual vector.
   * Useful for API compatibility but less memory-efficient.
   * Uses std::iota for efficient population.
   */
  // https://stackoverflow.com/questions/18625223
  inline std::vector< int >
  ToVector()
  {
    int start = i0;
    std::vector< int > v( n );
    std::iota( v.begin(), v.end(), start );
    return v;
  }
};

typedef Sequence NodeSeq; /**< Alias for Sequence, commonly used for node indices */

/**
 * @class RemoteNodeSeq
 * @brief Represents a sequence of nodes on a remote MPI host
 *
 * This class extends NodeSeq to include host information for
 * distributed simulations across multiple MPI processes. It's
 * used when creating or connecting neurons on remote hosts.
 *
 * MPI Distribution:
 * - Each host manages a subset of neurons
 * - Remote node sequences allow cross-host connections
 * - Used in multi-GPU and multi-node simulations
 *
 * Usage Example:
 * @code
 * // Create 100 neurons on host 2
 * RemoteNodeSeq remote_nodes(2, NodeSeq(0, 100));
 * // Connect local neurons to remote neurons
 * nestgpu.Connect(local_neurons, remote_nodes, conn_spec, syn_spec);
 * @endcode
 *
 * @see NESTGPU::RemoteCreate()
 * @see NESTGPU::RemoteConnect()
 */
class RemoteNodeSeq
{
public:
  int i_host;           /**< Index of the MPI host containing these nodes */
  NodeSeq node_seq;     /**< Sequence of node indices on the remote host */

  /**
   * @brief Constructor for RemoteNodeSeq
   * @param i_host Index of the remote host (default: 0)
   * @param node_seq Node sequence on the remote host (default: empty)
   */
  RemoteNodeSeq( int i_host = 0, NodeSeq node_seq = NodeSeq( 0, 0 ) )
    : i_host( i_host )
    , node_seq( node_seq )
  {
  }
};

/**
 * @enum ExceptionHandlingMode
 * @brief Defines exception handling behavior for NESTGPU operations
 *
 * Controls how NESTGPU handles errors and exceptions during simulation.
 * Allows either immediate exit or graceful error handling.
 */
enum
{
  ON_EXCEPTION_EXIT = 0,    /**< Exit immediately on exception */
  ON_EXCEPTION_HANDLE       /**< Handle exception gracefully and set error flags */
};

/**
 * @class NESTGPU
 * @brief Main GPU-accelerated spiking neural network simulation engine
 *
 * The NESTGPU class provides the core functionality for simulating large-scale
 * spiking neural networks on GPUs. It offers a Python-compatible interface
 * while optimizing performance through GPU acceleration.
 *
 * Architecture Overview:
 * - GPU-based neuron model simulation
 * - Efficient spike communication system
 * - Multi-GPU and MPI support for distributed computing
 * - Flexible connection rules and synapse models
 * - Real-time parameter access and modification
 *
 * Key Components:
 * - Node Groups: Organized collections of neurons of the same type
 * - Connection Management: Flexible connectivity rules and structures
 * - Spike Buffers: Efficient spike storage and delivery
 * - Random Number Generation: GPU-accelerated stochastic processes
 * - MPI Interface: Distributed simulation support
 *
 * Typical Usage Pattern:
 * 1. Initialize: NESTGPU ngpu; ngpu.SetRandomSeed(seed);
 * 2. Configure: ngpu.SetTimeResolution(0.1);
 * 3. Create Network: NodeSeq neurons = ngpu.Create("iaf_psc_exp", 1000);
 * 4. Connect: ngpu.Connect(neurons, neurons, conn_spec, syn_spec);
 * 5. Calibrate: ngpu.Calibrate();
 * 6. Simulate: ngpu.Simulate(1000.0);
 *
 * Performance Characteristics:
 * - Optimized for networks with >1000 neurons
 * - Near-linear scaling with network size
 * - Efficient spike communication patterns
 * - Memory bandwidth optimized data structures
 *
 * Thread Safety:
 * - Not thread-safe; use single instance per thread
 * - MPI processes should have separate instances
 *
 * @see BaseNeuron Base class for neuron implementations
 * @see Connection Connection management system
 * @see ConnSpec Connection rule specifications
 * @see SynSpec Synapse model specifications
 */
class NESTGPU
{
  /* Simulation Parameters */
  float time_resolution_; /**< Simulation timestep resolution in milliseconds */

  /* Random Number Generation */
  curandGenerator_t* random_generator_; /**< CUDA random number generator pointer */
  unsigned long long kernel_seed_;         /**< Seed for GPU random number generation */

  /* State Flags */
  bool calibrate_flag_; /**< Becomes true after network calibration is complete */
  bool create_flag_;    /**< Becomes true just before creation of the first node */

  /* Core Components */
  // Pointer to the connection object. Note that conn_ is of the type
  // pointer-to-the(abstract)-base class
  // while the object it will point to should be an instance of a derived class
  Connection* conn_;           /**< Connection management system (polymorphic) */
  Distribution* distribution_; /**< Random distribution system for parameters */
  Multimeter* multimeter_;     /**< Multi-variable recording device */
  int conn_struct_type_;       /**< Type of connection storage structure used */

  std::vector< BaseNeuron* > node_vect_;     /**< Vector of neuron model pointers */
  std::vector< SynModel* > syn_group_vect_; /**< Vector of synapse model pointers */

  /* MPI Configuration */
  int this_host_;          /**< Index of this MPI host (0-based) */
  int n_hosts_;            /**< Total number of MPI hosts */

  /* Communication Flags */
  bool external_spike_flag_; /**< If true, enables cross-host spike communication */
  bool mpi_flag_;           /**< True if MPI is initialized and active */
  bool mpi_bitpack_;        /**< Enable bit-packing for MPI spike communication */
  bool max_n_ports_warning_; /**< Enable warnings for maximum port limit */
  bool remote_spike_mul_;    /**< Enable spike multiplication for remote hosts */

  /* Memory Management */
  std::vector< int16_t > node_group_map_; /**< Host-side node group mapping */
  int16_t* d_node_group_map_;             /**< Device-side node group mapping (GPU) */

  /* Spike Buffer Configuration */
  int max_spike_buffer_size_; /**< Maximum size of spike buffers */
  int max_spike_num_;         /**< Maximum number of spikes per timestep */
  int max_spike_per_host_;    /**< Maximum spikes per host per timestep */
  int max_remote_spike_num_; /**< Maximum remote spikes to receive */

  double max_spike_num_fact_;

  double max_spike_per_host_fact_;

  double max_remote_spike_num_fact_;

  double t_min_;

  double neural_time_; // Neural activity time

  double sim_time_; // Simulation time in ms

  /* Simulation Time Management */
  double max_spike_num_fact_;        /**< Factor for computing max_spike_num_ */
  double max_spike_per_host_fact_;   /**< Factor for computing max_spike_per_host_ */
  double max_remote_spike_num_fact_; /**< Factor for computing max_remote_spike_num_ */
  double t_min_;                     /**< Minimum simulation time */

  double neural_time_;   /**< Neural activity simulation time (ms) */
  double sim_time_;      /**< Total simulation time (ms) */
  double neur_t0_;       /**< Neural activity simulation time origin (ms) */

  long long it_;         /**< Current simulation time index */
  long long Nt_;         /**< Total number of simulation time steps */

  /* Node Distribution */
  std::vector< int > n_remote_nodes_; /**< Number of remote nodes per host */

  /* Performance Timing */
  double start_real_time_; /**< Wall clock time when simulation started */
  double build_real_time_;  /**< Wall clock time when network was built */
  double end_real_time_;    /**< Wall clock time when simulation ended */

  /* Error Handling */
  bool error_flag_;        /**< True if an error has occurred */
  std::string error_message_; /**< Description of the last error */
  unsigned char error_code_;  /**< Error code for the last error */
  int on_exception_;        /**< Exception handling mode (ON_EXCEPTION_EXIT/HANDLE) */

  /* Configuration Parameters */
  int verbosity_level_;     /**< Level of detail for status messages (0=quiet) */
  bool print_time_;         /**< If true, print timing information */
  bool remove_conn_key_;    /**< Connection key removal flag */
  int nested_loop_algo_;    /**< Algorithm for nested loop operations */
  int spike_buffer_algo_;   /**< Algorithm for spike buffer management */
  bool check_node_maps_;    /**< Enable node map validation checks */

  /* Internal State */
  bool first_out_conn_in_device_; /**< First output connection in device flag */
  bool have_n_out_conn_;           /**< Have number of output connections flag */
  bool delete_remote_node_map_;    /**< Delete remote node map flag */
  bool delete_image_node_map_;     /**< Delete image node map flag */

  float use_all_source_node_fact_; /**< Factor for using all source nodes */

  std::vector< int > ext_neuron_input_spike_node_; /**< External neuron input spike nodes */
  std::vector< int > ext_neuron_input_spike_port_; /**< External neuron input spike ports */
  std::vector< float > ext_neuron_input_spike_mul_; /**< External neuron input spike multipliers */

  uint CreateNodeGroup( uint n_nodes, int n_ports );

  int CheckUncalibrated( std::string message );

  double* InitGetSpikeArray( uint n_nodes, int n_ports );

  int NodeGroupArrayInit();

  int ClearGetSpikeArrays();

  int FreeGetSpikeArrays();

  int FreeNodeGroupMap();

  uint CheckImageNodes( uint n_nodes );

  NodeSeq _Create( std::string model_name, uint n_nodes, int n_ports );

  double SpikeBufferUpdate_time_;

  double poisson_generator_time_;

  double neuron_Update_time_;

  double copy_ext_spike_time_;

  double organizeExternalSpike_time_;

  double SendSpikeToRemote_time_;

  double RecvSpikeFromRemote_time_;
  
  double CopySpikeFromRemote_time_;

  double DeliverSpikesToInputBuffers_time_;
  
  double NestedLoop_time_;

  double GetSpike_time_;

  double SpikeReset_time_;

  double ExternalSpikeReset_time_;

  double SendSpikeToRemote_comm_time_;

  double RecvSpikeFromRemote_comm_time_;

  double SendSpikeToRemote_CUDAcp_time_;

  double RecvSpikeFromRemote_CUDAcp_time_;

  double MpiBitPack_time_;

  double MpiBitUnpack_time_;

  int64_t SpikeNumAllgather_send_;
  int64_t SpikeNumAllgather_send_packed_;
  int64_t SpikeNumAllgather_recv_;
  int64_t SpikeNumAllgather_recv_packed_;
  
  bool first_simulation_flag_;

public:
  /**
   * @brief Constructor for NESTGPU
   *
   * Initializes the NESTGPU simulation engine with default parameters.
   * Sets up CUDA context, allocates GPU memory, and initializes internal
   * data structures for network simulation.
   *
   * Initialization includes:
   * - CUDA device initialization
   * - Random number generator setup
   * - Memory allocation for internal structures
   * - Default parameter configuration
   *
   * @note Must be called before any other NESTGPU methods
   * @warning Throws exceptions if CUDA initialization fails
   */
  NESTGPU();

  /**
   * @brief Destructor for NESTGPU
   *
   * Cleans up all GPU memory and resources allocated during the lifetime
   * of the NESTGPU instance. This includes:
   * - Freeing GPU memory arrays
   * - Destroying CUDA random number generators
   * - Closing MPI connections if active
   * - Releasing internal data structures
   *
   * @note It's safe to call even if simulation was never run
   */
  ~NESTGPU();

  /**
   * @brief Set the number of MPI hosts for distributed simulation
   * @param n_hosts Total number of MPI processes/hosts
   * @return 0 on success, error code on failure
   *
   * This method configures the simulation for distributed computing across
   * multiple MPI processes. Each host manages a subset of neurons.
   *
   * Usage:
   * @code
   * ngpu.setNHosts(4);  // 4-way distributed simulation
   * @endcode
   *
   * @note Must be called before network creation
   * @warning Requires MPI initialization
   * @see ConnectMpiInit()
   */
  int setNHosts( int n_hosts );

  /**
   * @brief Set the index of the current MPI host
   * @param i_host Index of this host (0 to n_hosts-1)
   * @return 0 on success, error code on failure
   *
   * Identifies which host this process is in the distributed simulation.
   * Each host should have a unique index.
   *
   * @note Must be called after setNHosts() and before network creation
   * @see setNHosts()
   */
  int setThisHost( int i_host );

  /**
   * @brief Set the random seed for all stochastic processes
   * @param seed Random seed value (unsigned long long)
   * @return 0 on success, error code on failure
   *
   * Initializes all random number generators (both host and GPU) with
   * the specified seed to ensure reproducible simulations.
   *
   * Random Processes Affected:
   * - Poisson spike generation
   * - Connection randomness
   * - Parameter randomization
   * - Neuron model stochasticity
   *
   * Usage:
   * @code
   * ngpu.SetRandomSeed(12345);  // Reproducible simulation
   * @endcode
   *
   * @note Should be called before network creation for reproducibility
   * @warning Different seeds produce different network patterns
   */
  int SetRandomSeed( unsigned long long seed );

  /**
   * @brief Set the simulation time resolution (timestep)
   * @param time_res Time resolution in milliseconds
   * @return 0 on success, error code on failure
   *
   * Sets the integration timestep for the simulation. Smaller values
   * provide better accuracy but slower performance.
   *
   * Typical Values:
   * - 0.1 ms: High accuracy, slower
   * - 1.0 ms: Standard accuracy
   * - 0.01 ms: Very high accuracy, much slower
   *
   * Considerations:
   * - Should be smaller than smallest synaptic delay
   * - Affects numerical stability of integration
   * - Impacts memory usage for spike recording
   *
   * Usage:
   * @code
   * ngpu.SetTimeResolution(0.1);  // 0.1 ms timestep
   * @endcode
   *
   * @note Must be called before Calibrate()
   * @warning Cannot be changed after simulation starts
   */
  int SetTimeResolution( float time_res );

  /**
   * @brief Get the current time resolution
   * @return Time resolution in milliseconds
   *
   * Returns the simulation timestep set by SetTimeResolution().
   */
  inline float
  GetTimeResolution()
  {
    return time_resolution_;
  }

  /**
   * @brief Set the total simulation time
   * @param sim_time Total simulation time in milliseconds
   * @return 0 on success
   *
   * Sets the duration for the next simulation run. This is used
   * by Simulate() to determine when to stop.
   *
   * Usage:
   * @code
   * ngpu.SetSimTime(1000.0);  // Simulate for 1 second (1000 ms)
   * ngpu.Simulate();           // Run for 1000 ms
   * @endcode
   *
   * @note Can be changed between simulation calls
   */
  inline int
  SetSimTime( float sim_time )
  {
    sim_time_ = sim_time;
    return 0;
  }

  /**
   * @brief Get the total simulation time
   * @return Total simulation time in milliseconds
   *
   * Returns the simulation time set by SetSimTime().
   */
  inline float
  GetSimTime()
  {
    return sim_time_;
  }

  /**
   * @brief Set the verbosity level for status messages
   * @param verbosity_level Verbosity level (0=quiet, higher=more verbose)
   * @return 0 on success
   *
   * Controls the amount of debugging and status information printed
   * during simulation.
   *
   * Levels:
   * - 0: Quiet mode (no output)
   * - 1-3: Basic status information
   * - 4-5: Detailed debugging information
   * - 6+: Very verbose (all operations)
   *
   * Usage:
   * @code
   * ngpu.SetVerbosityLevel(5);  // Detailed output
   * @endcode
   */
  inline int
  SetVerbosityLevel( int verbosity_level )
  {
    verbosity_level_ = verbosity_level;
    verbose_print_ns::verbosity_level_ = verbosity_level;
    return 0;
  }

  int SetNestedLoopAlgo( int nested_loop_algo );

  inline int
  SetPrintTime( bool print_time )
  {
    print_time_ = print_time;
    return 0;
  }

  inline int
  SetCheckNodeMaps( bool check_node_maps )
  {
    check_node_maps_ = check_node_maps;
    return 0;
  }

  int SetMaxSpikeBufferSize( int max_size );
  int GetMaxSpikeBufferSize();

  uint GetNLocalNodes();

  uint GetNTotalNodes();

  int setConnStructType( int conn_struct_type );

  int
  HostNum()
  {
    return n_hosts_;
  }

  int
  HostId()
  {
    return this_host_;
  }

  std::string HostIdStr();

  size_t getCUDAMemHostUsed();

  size_t getCUDAMemHostPeak();

  size_t getCUDAMemTotal();

  size_t getCUDAMemFree();

  int GetNBoolParam();
  std::vector< std::string > GetBoolParamNames();
  bool IsBoolParam( std::string param_name );
  int GetBoolParamIdx( std::string param_name );
  bool GetBoolParam( std::string param_name );
  int SetBoolParam( std::string param_name, bool val );

  int GetNFloatParam();
  std::vector< std::string > GetFloatParamNames();
  bool IsFloatParam( std::string param_name );
  int GetFloatParamIdx( std::string param_name );
  float GetFloatParam( std::string param_name );
  int SetFloatParam( std::string param_name, float val );

  int GetNIntParam();
  std::vector< std::string > GetIntParamNames();
  bool IsIntParam( std::string param_name );
  int GetIntParamIdx( std::string param_name );
  int GetIntParam( std::string param_name );
  int SetIntParam( std::string param_name, int val );

  /**
   * @brief Create neurons of a specified model
   * @param model_name Name of the neuron model (e.g., "iaf_psc_exp", "aeif_cond_alpha")
   * @param n_nodes Number of neurons to create (default: 1)
   * @param n_ports Number of input ports for synaptic connections (default: 1)
   * @return NodeSeq representing the created neuron sequence
   *
   * Creates a group of neurons of the specified model with default parameters.
   * The neurons are assigned consecutive indices and can be referenced using
   * the returned NodeSeq.
   *
   * Available Neuron Models:
   * - "iaf_psc_exp": Leaky integrate-and-fire with exponential PSC
   * - "iaf_psc_alpha": LIF with alpha-function PSC
   * - "aeif_cond_alpha": Adaptive exponential integrate-and-fire with conductance
   * - "aeif_psc_exp": AdEx with current-based exponential PSC
   * - "izhikevich": Izhikevich model
   * - "poisson_generator": Poisson spike train generator
   * - "spike_generator": Deterministic spike pattern generator
   * - "spike_detector": Device for recording spike times
   * - "parrot_neuron": Relay neuron that repeats input spikes
   *
   * Usage Examples:
   * @code
   * // Create 100 LIF neurons
   * NodeSeq neurons = ngpu.Create("iaf_psc_exp", 100);
   *
   * // Create single neuron with 3 input ports
   * NodeSeq neuron = ngpu.Create("aeif_cond_alpha", 1, 3);
   * @endcode
   *
   * Performance:
   * - Efficient batch creation
   * - Memory allocated on GPU
   * - Optimized for large groups
   *
   * @note Must be called before Connect()
   * @warning Cannot create neurons after Calibrate()
   * @see SetNeuronParam() for setting individual neuron parameters
   */
  NodeSeq Create( std::string model_name, uint n_nodes = 1, int n_ports = 1 );

  /**
   * @brief Create neurons on a remote MPI host
   * @param i_host Index of the remote host
   * @param model_name Name of the neuron model
   * @param n_nodes Number of neurons to create (default: 1)
   * @param n_ports Number of input ports (default: 1)
   * @return RemoteNodeSeq representing the remote neuron sequence
   *
   * Creates neurons on a specific remote host in distributed simulations.
   * This allows cross-host connections and distributed network management.
   *
   * Usage:
   * @code
   * // Create 1000 neurons on host 2
   * RemoteNodeSeq remote_neurons = ngpu.RemoteCreate(2, "iaf_psc_exp", 1000);
   * @endcode
   *
   * @note Requires MPI initialization
   * @see ConnectMpiInit(), setNHosts()
   */
  RemoteNodeSeq RemoteCreate( int i_host, std::string model_name, inode_t n_nodes = 1, int n_ports = 1 );

  int CreateRecord( std::string file_name, std::string* var_name_arr, int* i_node_arr, int n_node );
  int CreateRecord( std::string file_name, std::string* var_name_arr, int* i_node_arr, int* port_arr, int n_node );
  std::vector< std::vector< float > >* GetRecordData( int i_record );

  int SetNeuronParam( int i_node, int n_neuron, std::string param_name, float val );

  int SetNeuronParam( int* i_node, int n_neuron, std::string param_name, float val );

  int SetNeuronParam( int i_node, int n_neuron, std::string param_name, float* param, int array_size );

  int SetNeuronParam( int* i_node, int n_neuron, std::string param_name, float* param, int array_size );

  int
  SetNeuronParam( NodeSeq nodes, std::string param_name, float val )
  {
    return SetNeuronParam( nodes.i0, nodes.n, param_name, val );
  }

  int
  SetNeuronParam( NodeSeq nodes, std::string param_name, float* param, int array_size )
  {
    return SetNeuronParam( nodes.i0, nodes.n, param_name, param, array_size );
  }

  int
  SetNeuronParam( std::vector< int > nodes, std::string param_name, float val )
  {
    return SetNeuronParam( nodes.data(), nodes.size(), param_name, val );
  }

  int
  SetNeuronParam( std::vector< int > nodes, std::string param_name, float* param, int array_size )
  {
    return SetNeuronParam( nodes.data(), nodes.size(), param_name, param, array_size );
  }

  int SetNeuronIntVar( int i_node, int n_neuron, std::string var_name, int val );

  int SetNeuronIntVar( int* i_node, int n_neuron, std::string var_name, int val );

  int
  SetNeuronIntVar( NodeSeq nodes, std::string var_name, int val )
  {
    return SetNeuronIntVar( nodes.i0, nodes.n, var_name, val );
  }

  int
  SetNeuronIntVar( std::vector< int > nodes, std::string var_name, int val )
  {
    return SetNeuronIntVar( nodes.data(), nodes.size(), var_name, val );
  }

  int SetNeuronVar( int i_node, int n_neuron, std::string var_name, float val );

  int SetNeuronVar( int* i_node, int n_neuron, std::string var_name, float val );

  int SetNeuronVar( int i_node, int n_neuron, std::string var_name, float* var, int array_size );

  int SetNeuronVar( int* i_node, int n_neuron, std::string var_name, float* var, int array_size );

  int
  SetNeuronVar( NodeSeq nodes, std::string var_name, float val )
  {
    return SetNeuronVar( nodes.i0, nodes.n, var_name, val );
  }

  int
  SetNeuronVar( NodeSeq nodes, std::string var_name, float* var, int array_size )
  {
    return SetNeuronVar( nodes.i0, nodes.n, var_name, var, array_size );
  }

  int
  SetNeuronVar( std::vector< int > nodes, std::string var_name, float val )
  {
    return SetNeuronVar( nodes.data(), nodes.size(), var_name, val );
  }

  int
  SetNeuronVar( std::vector< int > nodes, std::string var_name, float* var, int array_size )
  {
    return SetNeuronVar( nodes.data(), nodes.size(), var_name, var, array_size );
  }

  ////////////////////////////////////////////////////////////////////////

  int SetNeuronScalParamDistr( int i_node, int n_node, std::string param_name );

  int SetNeuronScalVarDistr( int i_node, int n_node, std::string var_name );

  int SetNeuronPortParamDistr( int i_node, int n_node, std::string param_name );

  int SetNeuronPortVarDistr( int i_node, int n_node, std::string var_name );

  int SetNeuronPtScalParamDistr( int* i_node, int n_node, std::string param_name );

  int SetNeuronPtScalVarDistr( int* i_node, int n_node, std::string var_name );

  int SetNeuronPtPortParamDistr( int* i_node, int n_node, std::string param_name );

  int SetNeuronPtPortVarDistr( int* i_node, int n_node, std::string var_name );

  int SetDistributionIntParam( std::string param_name, int val );

  int SetDistributionScalParam( std::string param_name, float val );

  int SetDistributionVectParam( std::string param_name, float val, int i );

  int SetDistributionFloatPtParam( std::string param_name, float* array_pt );

  int IsDistributionFloatParam( std::string param_name );

  ////////////////////////////////////////////////////////////////////////

  int GetNeuronParamSize( int i_node, std::string param_name );

  int GetNeuronVarSize( int i_node, std::string var_name );

  float* GetNeuronParam( int i_node, int n_neuron, std::string param_name );

  float* GetNeuronParam( int* i_node, int n_neuron, std::string param_name );

  float*
  GetNeuronParam( NodeSeq nodes, std::string param_name )
  {
    return GetNeuronParam( nodes.i0, nodes.n, param_name );
  }

  float*
  GetNeuronParam( std::vector< int > nodes, std::string param_name )
  {
    return GetNeuronParam( nodes.data(), nodes.size(), param_name );
  }

  float* GetArrayParam( int i_node, std::string param_name );

  int* GetNeuronIntVar( int i_node, int n_neuron, std::string var_name );

  int* GetNeuronIntVar( int* i_node, int n_neuron, std::string var_name );

  int*
  GetNeuronIntVar( NodeSeq nodes, std::string var_name )
  {
    return GetNeuronIntVar( nodes.i0, nodes.n, var_name );
  }

  int*
  GetNeuronIntVar( std::vector< int > nodes, std::string var_name )
  {
    return GetNeuronIntVar( nodes.data(), nodes.size(), var_name );
  }

  float* GetNeuronVar( int i_node, int n_neuron, std::string var_name );

  float* GetNeuronVar( int* i_node, int n_neuron, std::string var_name );

  float*
  GetNeuronVar( NodeSeq nodes, std::string var_name )
  {
    return GetNeuronVar( nodes.i0, nodes.n, var_name );
  }

  float*
  GetNeuronVar( std::vector< int > nodes, std::string var_name )
  {
    return GetNeuronVar( nodes.data(), nodes.size(), var_name );
  }

  float* GetArrayVar( int i_node, std::string param_name );

  int GetNodeSequenceOffset( int i_node, int n_node, int& i_group );

  std::vector< int > GetNodeArrayWithOffset( int* i_node, int n_node, int& i_group );

  int IsNeuronScalParam( int i_node, std::string param_name );

  int IsNeuronPortParam( int i_node, std::string param_name );

  int IsNeuronArrayParam( int i_node, std::string param_name );

  int IsNeuronIntVar( int i_node, std::string var_name );

  int IsNeuronScalVar( int i_node, std::string var_name );

  int IsNeuronPortVar( int i_node, std::string var_name );

  int IsNeuronArrayVar( int i_node, std::string var_name );

  int SetSpikeGenerator( int i_node, int n_spikes, float* spike_time, float* spike_mul );

  /**
   * @brief Calibrate the network before simulation
   * @return 0 on success, error code on failure
   *
   * Prepares the network for simulation by:
   * - Allocating GPU memory for spike buffers
   * - Building connection data structures
   * - Precomputing connection delays
   * - Initializing neuron states
   * - Setting up random number generators
   * - Configuring MPI communication if enabled
   *
   * This method must be called once after network creation and connection
   * setup, but before running the first simulation.
   *
   * Typical Workflow:
   * @code
   * ngpu.SetTimeResolution(0.1);
   * NodeSeq neurons = ngpu.Create("iaf_psc_exp", 1000);
   * ngpu.Connect(neurons, neurons, conn_spec, syn_spec);
   * ngpu.Calibrate();  // Prepare for simulation
   * ngpu.Simulate(1000.0);
   * @endcode
   *
   * Performance:
   * - One-time overhead before simulation
   * - Enables efficient spike communication
   * - Optimizes memory access patterns
   *
   * @note Must be called after all Create() and Connect() calls
   * @note Must be called before first Simulate() call
   * @warning Cannot add neurons or connections after calibration
   */
  int Calibrate();

  /**
   * @brief Run simulation for the time set by SetSimTime()
   * @return 0 on success, error code on failure
   *
   * Executes the neural network simulation for the duration specified
   * by SetSimTime(). This is the main simulation method that advances
   * the network state in time.
   *
   * Simulation Process:
   * 1. Generate Poisson spikes (if any generators)
   * 2. Deliver spikes to neuron inputs
   * 3. Update neuron states (integrate equations)
   * 4. Check for spike generation
   * 5. Record spikes (if detectors enabled)
   * 6. Handle MPI communication (if distributed)
   * 7. Repeat until simulation time is reached
   *
   * Usage:
   * @code
   * ngpu.SetSimTime(1000.0);  // Simulate 1 second
   * ngpu.Simulate();           // Run simulation
   * @endcode
   *
   * @note Requires prior call to Calibrate()
   * @see Simulate(float) for alternative interface
   */
  int Simulate();

  /**
   * @brief Run simulation for specified duration
   * @param sim_time Simulation duration in milliseconds
   * @return 0 on success, error code on failure
   *
   * Convenience method that combines SetSimTime() and Simulate().
   * Runs the simulation for the specified duration.
   *
   * Usage:
   * @code
   * ngpu.Simulate(1000.0);  // Simulate for 1000 ms (1 second)
   * ngpu.Simulate(500.0);   // Simulate for additional 500 ms
   * @endcode
   *
   * Performance Considerations:
   * - Time scales linearly with sim_time and network size
   * - GPU utilization typically >80% for large networks
   * - Memory bandwidth is often the bottleneck
   *
   * @note Requires prior call to Calibrate()
   * @note Can be called multiple times for successive simulation periods
   */
  int Simulate( float sim_time );

  /**
   * @brief Start simulation in stepping mode
   * @return 0 on success, error code on failure
   *
   * Initializes simulation for step-by-step execution using SimulationStep().
   * Useful for:
   * - Real-time simulations with external input
   * - Interactive simulations
   * - Debugging and analysis
   *
   * Usage:
   * @code
   * ngpu.StartSimulation();
   * for (int i = 0; i < n_steps; i++) {
   *     ngpu.SimulationStep();
   *     // Perform operations between steps
   * }
   * @endcode
   *
   * @see SimulationStep()
   */
  int StartSimulation();

  /**
   * @brief Execute single simulation step
   * @return 0 on success, error code on failure
   *
   * Advances the simulation by one timestep (time_resolution_).
   * Used with StartSimulation() for fine-grained control.
   *
   * @note Requires prior call to StartSimulation()
   * @see StartSimulation()
   */
  int SimulationStep();

  /**
   * @brief Print performance timing information
   * @param verbosity_level Detail level (0-10, default: 5)
   * @return 0 on success
   *
   * Outputs detailed timing statistics for various simulation components,
   * useful for performance analysis and optimization.
   *
   * Information Printed:
   * - Total simulation time
   * - Time spent in each major component
   * - GPU memory usage
   * - MPI communication statistics (if applicable)
   * - Spike buffer statistics
   *
   * Usage:
   * @code
   * ngpu.Simulate(1000.0);
   * ngpu.PrintTimers(5);  // Print timing information
   * @endcode
   *
   * @note Most useful after simulation has run
   */
  int PrintTimers(int verbosity_level = 5);

  int ConnectMpiInit( int argc, char** argv );

  int FakeConnectMpiInit(int n_hosts, int this_host);

  int MpiFinalize();

  void
  SetErrorFlag( bool error_flag )
  {
    error_flag_ = error_flag;
  }

  void
  SetErrorMessage( std::string error_message )
  {
    error_message_ = error_message;
  }

  void
  SetErrorCode( unsigned char error_code )
  {
    error_code_ = error_code;
  }

  void
  SetOnException( int on_exception )
  {
    on_exception_ = on_exception;
  }

  bool
  GetErrorFlag()
  {
    return error_flag_;
  }

  char*
  GetErrorMessage()
  {
    return &error_message_[ 0 ];
  }

  unsigned char
  GetErrorCode()
  {
    return error_code_;
  }

  int
  OnException()
  {
    return on_exception_;
  }

  unsigned int* RandomInt( size_t n );

  float* RandomUniform( size_t n );

  float* RandomNormal( size_t n, float mean, float stddev );

  float* RandomNormalClipped( size_t n, float mean, float stddev, float vmin, float vmax, float vstep );

  int Connect( inode_t i_source,
    inode_t n_source,
    inode_t i_target,
    inode_t n_target,
    ConnSpec& conn_spec,
    SynSpec& syn_spec );

  int Connect( inode_t i_source,
    inode_t n_source,
    inode_t* target,
    inode_t n_target,
    ConnSpec& conn_spec,
    SynSpec& syn_spec );

  int Connect( inode_t* source,
    inode_t n_source,
    inode_t i_target,
    inode_t n_target,
    ConnSpec& conn_spec,
    SynSpec& syn_spec );

  int Connect( inode_t* source,
    inode_t n_source,
    inode_t* target,
    inode_t n_target,
    ConnSpec& conn_spec,
    SynSpec& syn_spec );

  int Connect( NodeSeq source, NodeSeq target, ConnSpec& conn_spec, SynSpec& syn_spec );

  int Connect( NodeSeq source, std::vector< inode_t > target, ConnSpec& conn_spec, SynSpec& syn_spec );

  int Connect( std::vector< inode_t > source, NodeSeq target, ConnSpec& conn_spec, SynSpec& syn_spec );

  int Connect( std::vector< inode_t > source, std::vector< inode_t > target, ConnSpec& conn_spec, SynSpec& syn_spec );

  int RemoteConnect( int i_source_host,
    inode_t i_source,
    inode_t n_source,
    int i_target_host,
    inode_t i_target,
    inode_t n_target,
    int i_host_group,
    ConnSpec& conn_spec,
    SynSpec& syn_spec );

  int RemoteConnect( int i_source_host,
    inode_t i_source,
    inode_t n_source,
    int i_target_host,
    inode_t* target,
    inode_t n_target,
    int i_host_group,
    ConnSpec& conn_spec,
    SynSpec& syn_spec );

  int RemoteConnect( int i_source_host,
    inode_t* source,
    inode_t n_source,
    int i_target_host,
    inode_t i_target,
    inode_t n_target,
    int i_host_group,
    ConnSpec& conn_spec,
    SynSpec& syn_spec );

  int RemoteConnect( int i_source_host,
    inode_t* source,
    inode_t n_source,
    int i_target_host,
    inode_t* target,
    inode_t n_target,
    int i_host_group,
    ConnSpec& conn_spec,
    SynSpec& syn_spec );

  int RemoteConnect( int i_source_host,
    NodeSeq source,
    int i_target_host,
    NodeSeq target,
    int i_host_group,
    ConnSpec& conn_spec,
    SynSpec& syn_spec );

  int RemoteConnect( int i_source_host,
    NodeSeq source,
    int i_target_host,
    std::vector< inode_t > target,
    int i_host_group,
    ConnSpec& conn_spec,
    SynSpec& syn_spec );

  int RemoteConnect( int i_source_host,
    std::vector< inode_t > source,
    int i_target_host,
    NodeSeq target,
    int i_host_group,
    ConnSpec& conn_spec,
    SynSpec& syn_spec );

  int RemoteConnect( int i_source_host,
    std::vector< inode_t > source,
    int i_target_host,
    std::vector< inode_t > target,
    int i_host_group,
    ConnSpec& conn_spec,
    SynSpec& syn_spec );
  
  // Method that creates a group of hosts for remote spike communication (i.e. a group of MPI processes)
  // host_arr: array of host inexes, n_hosts: nomber of hosts in the group
  int CreateHostGroup(int *host_arr, int n_hosts);
  
  std::vector< std::string > GetScalVarNames( int i_node );

  int GetNIntVar( int i_node );

  std::vector< std::string > GetIntVarNames( int i_node );

  int GetNScalVar( int i_node );

  std::vector< std::string > GetPortVarNames( int i_node );

  int GetNPortVar( int i_node );

  std::vector< std::string > GetScalParamNames( int i_node );

  int GetNScalParam( int i_node );

  std::vector< std::string > GetPortParamNames( int i_node );

  int GetNPortParam( int i_node );

  std::vector< std::string > GetArrayParamNames( int i_node );

  int GetNArrayParam( int i_node );

  std::vector< std::string > GetArrayVarNames( int i_node );

  std::vector< std::string > GetGroupParamNames( int i_node );

  int GetNGroupParam( int i_node );

  int GetNArrayVar( int i_node );

  int GetConnectionFloatParamIndex( std::string param_name );

  int GetConnectionIntParamIndex( std::string param_name );

  int IsConnectionFloatParam( std::string param_name );

  int IsConnectionIntParam( std::string param_name );

  int GetConnectionFloatParam( int64_t* conn_ids, int64_t n_conn, float* h_param_arr, std::string param_name );

  int GetConnectionIntParam( int64_t* conn_ids, int64_t n_conn, int* h_param_arr, std::string param_name );

  int SetConnectionFloatParamDistr( int64_t* conn_ids, int64_t n_conn, std::string param_name );

  int SetConnectionFloatParam( int64_t* conn_ids, int64_t n_conn, float val, std::string param_name );

  int SetConnectionIntParamArr( int64_t* conn_ids, int64_t n_conn, int* h_param_arr, std::string param_name );

  int SetConnectionIntParam( int64_t* conn_ids, int64_t n_conn, int val, std::string param_name );

  int GetConnectionStatus( int64_t* conn_ids,
    int64_t n_conn,
    inode_t* source,
    inode_t* target,
    int* port,
    int* syn_group,
    float* delay,
    float* weight );

  int64_t* GetConnections( inode_t i_source,
    inode_t n_source,
    inode_t i_target,
    inode_t n_target,
    int syn_group,
    int64_t* n_conn );

  int64_t* GetConnections( inode_t* i_source_pt,
    inode_t n_source,
    inode_t i_target,
    inode_t n_target,
    int syn_group,
    int64_t* n_conn );

  int64_t* GetConnections( inode_t i_source,
    inode_t n_source,
    inode_t* i_target_pt,
    inode_t n_target,
    int syn_group,
    int64_t* n_conn );

  int64_t* GetConnections( inode_t* i_source_pt,
    inode_t n_source,
    inode_t* i_target_pt,
    inode_t n_target,
    int syn_group,
    int64_t* n_conn );

  int64_t* GetConnections( NodeSeq source, NodeSeq target, int syn_group, int64_t* n_conn );

  int64_t* GetConnections( std::vector< inode_t > source, NodeSeq target, int syn_group, int64_t* n_conn );

  int64_t* GetConnections( NodeSeq source, std::vector< inode_t > target, int syn_group, int64_t* n_conn );

  int64_t*
  GetConnections( std::vector< inode_t > source, std::vector< inode_t > target, int syn_group, int64_t* n_conn );

  int CreateSynGroup( std::string model_name );

  int GetSynGroupNParam( int syn_group );

  std::vector< std::string > GetSynGroupParamNames( int syn_group );

  bool IsSynGroupParam( int syn_group, std::string param_name );

  int GetSynGroupParamIdx( int syn_group, std::string param_name );

  float GetSynGroupParam( int syn_group, std::string param_name );

  int SetSynGroupParam( int syn_group, std::string param_name, float val );

  int SynGroupCalibrate();

  int ActivateSpikeCount( int i_node, int n_node );

  int
  ActivateSpikeCount( NodeSeq nodes )
  {
    return ActivateSpikeCount( nodes.i0, nodes.n );
  }

  int ActivateRecSpikeTimes( int i_node, int n_node, int max_n_rec_spike_times );

  int
  ActivateRecSpikeTimes( NodeSeq nodes, int max_n_rec_spike_times )
  {
    return ActivateRecSpikeTimes( nodes.i0, nodes.n, max_n_rec_spike_times );
  }

  int SetRecSpikeTimesStep( int i_node, int n_node, int rec_spike_times_step );

  int
  SetRecSpikeTimesStep( NodeSeq nodes, int rec_spike_times_step )
  {
    return SetRecSpikeTimesStep( nodes.i0, nodes.n, rec_spike_times_step );
  }

  int GetNRecSpikeTimes( int i_node );

  int GetRecSpikeTimes( int i_node, int n_node, int** n_spike_times_pt, float*** spike_times_pt );

  int
  GetRecSpikeTimes( NodeSeq nodes, int** n_spike_times_pt, float*** spike_times_pt )
  {
    return GetRecSpikeTimes( nodes.i0, nodes.n, n_spike_times_pt, spike_times_pt );
  }

  int PushSpikesToNodes( int n_spikes, int* node_id, float* spike_mul );

  int PushSpikesToNodes( int n_spikes, int* node_id );

  int GetExtNeuronInputSpikes( int* n_spikes, int** node, int** port, float** spike_mul, bool include_zeros );

  int SetNeuronGroupParam( int i_node, int n_node, std::string param_name, float val );

  int IsNeuronGroupParam( int i_node, std::string param_name );

  float GetNeuronGroupParam( int i_node, std::string param_name );

  int ExternalSpikeInit();

  int ExternalSpikeReset();

  int CopySpikeFromRemote();

  int SendSpikeToRemote( int n_ext_spikes );

  int RecvSpikeFromRemote();

  int organizeExternalSpikes( int n_ext_spikes );

  ///////////////////////////////////////////////////////////////////////////////////////////////////////
  // Build connections with fixed indegree rule for source neurons and target neurons distributed across
  // MPI processes (hosts)
  // Case with both source and target nodes contiguous, represented by starting index and number of nodes 
  ///////////////////////////////////////////////////////////////////////////////////////////////////////
  int ConnectDistributedFixedIndegree
  (int *source_host_arr, int n_source_host, inode_t *source_arr, inode_t *n_source_arr,
   int *target_host_arr, int n_target_host, inode_t *target_arr, inode_t *n_target_arr,
   int indegree, int i_host_group, SynSpec &syn_spec);
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////
  // Build connections with fixed indegree rule for source neurons and target neurons distributed across
  // MPI processes (hosts)
  // Case with source nodes stored in an array,
  // target nodes contiguous, represented by starting index and number of nodes 
  ///////////////////////////////////////////////////////////////////////////////////////////////////////
  int ConnectDistributedFixedIndegree
  (int *source_host_arr, int n_source_host, inode_t **source_arr, inode_t *n_source_arr,
   int *target_host_arr, int n_target_host, inode_t *target_arr, inode_t *n_target_arr,
   int indegree, int i_host_group, SynSpec &syn_spec);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////
  // Build connections with fixed indegree rule for source neurons and target neurons distributed across
  // MPI processes (hosts)
  // Case with source nodes contiguous, represented by starting index and number of nodes,
  // target nodes stored in an array
  ///////////////////////////////////////////////////////////////////////////////////////////////////////
  int ConnectDistributedFixedIndegree
  (int *source_host_arr, int n_source_host, inode_t *source_arr, inode_t *n_source_arr,
   int *target_host_arr, int n_target_host, inode_t **target_arr, inode_t *n_target_arr,
   int indegree, int i_host_group, SynSpec &syn_spec);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////
  // Build connections with fixed indegree rule for source neurons and target neurons distributed across
  // MPI processes (hosts)
  // Case with both source nodes and target nodes stored in arrays
  ///////////////////////////////////////////////////////////////////////////////////////////////////////
  int ConnectDistributedFixedIndegree
  (int *source_host_arr, int n_source_host, inode_t **source_arr, inode_t *n_source_arr,
   int *target_host_arr, int n_target_host, inode_t **target_arr, inode_t *n_target_arr,
   int indegree, int i_host_group, SynSpec &syn_spec);

  
};

#endif
