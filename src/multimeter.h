/*
 *  multimeter.h
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
 * @file multimeter.h
 * @brief Continuous variable recording device implementation
 *
 * This file implements the multimeter device, which records continuous
 * time-series data from neuron state variables during simulation. It provides
 * the primary mechanism for monitoring neural dynamics beyond spike times.
 *
 * Technical Architecture:
 * The multimeter operates as a recording system that samples neuron
 * variables at each timestep and stores them for analysis. Unlike spike
 * detectors, it captures the continuous evolution of neural state variables.
 *
 * Recording Capabilities:
 * - Membrane potential (V_m)
 * - Synaptic conductances (g_ex, g_in)
 * - Adaptation currents (w)
 * - Any state variable defined in neuron models
 * - Multiple variables simultaneously
 * - Multiple neurons concurrently
 *
 * Data Flow:
 * 1. During simulation: Sample neuron states at each timestep
 * 2. Store in GPU memory buffers
 * 3. Transfer to host memory after simulation
 * 4. Access via GetRecordData() interface
 *
 * Memory Management:
 * - Pre-allocated buffers based on simulation duration
 * - GPU memory for high-speed recording
 * - Host memory for user access
 * - Structured data organization
 *
 * GPU Implementation:
 * - Coalesced memory access for efficiency
 * - Minimal computational overhead
 * - Parallel recording from multiple neurons
 * - Efficient memory bandwidth utilization
 *
 * Data Organization:
 * - 2D array: [timestep × (1 + n_neurons)]
 * - First column: simulation time
 * - Remaining columns: neuron variable values
 * - Accessible as Python lists or NumPy arrays
 *
 * Performance Characteristics:
 * - Memory bandwidth limited
 * - Scales with: simulation_time × n_neurons × n_variables
 * - Efficient for moderate recording durations
 * - May impact performance with many recorded neurons
 *
 * Technical Limitations:
 * - Sampling interval fixed to simulation resolution
 * - Memory usage scales linearly with simulation time
 * - Large recordings may exhaust GPU memory
 * - No downsampling or filtering
 *
 * Usage Considerations:
 * - Record only necessary variables
 * - Limit recording duration for large networks
 * - Consider recording subsets of neurons
 * - Balance between detail and memory usage
 *
 * Integration Points:
 * - BaseNeuron: Access to neuron state variables
 * - Recording system: Part of data framework
 * - Python interface: Data export to Python
 * - File I/O: Optional direct file writing
 *
 * Applications:
 * - Neural dynamics analysis
 * - Phase plane analysis
 * - Network synchronization studies
 * - Computational neuroscience research
 * - Model validation and debugging
 *
 * Implementation Details:
 * - Record class: Manages recording metadata
 * - Multimeter class: Core recording functionality
 * - Variable access: Through neuron model interfaces
 * - Data export: Structured array format
 *
 * @see BaseNeuron Access to neuron state variables
 * @see spike_detector.h Spike event recording
 * @see nestgpu.h Recording interface functions
 */

#ifndef MULTIMETER_H
#define MULTIMETER_H
#include "base_neuron.h"
#include <stdio.h>
#include <string>
#include <vector>

/* BeginUserDocs: device, recorder

Short description
+++++++++++++++++

Sampling continuous quantities from neurons

Description
+++++++++++

The ``multimeter`` allows to record analog values from neurons.
Differently from NEST, a multimeter can be created using the command
``CreateRecord`` and takes as input the following parameters:

* a string representing the label of the record
* a list of strings with the name of the parameter to be recorded
* a list of node ids from the nodes to be recorded
* a list of integers indicating the node port to be recorded

Thus, the recording of the membrane potential for a single neuron can be
done as follows:

::

   recorder = nestgpu.CreateRecord('label', ['V_m'], [neuron[0]], [0]})

The lenght of the lists should be the same for all the three list
entries of ``CreateRecord``.

The sampling interval for recordings is the same one as the simulation
resolution (default 0.1 ms) and cannot be changed.

Differently from the NEST multimeter, the recorder should not be connected
with the nodes through a Connect routine since the nodes connected
to the record are specified in the ``CreateRecord`` routine.

The command ``GetRecordData`` returns, after the simulation, an
array containing the values of the parameters recorded for every node
specified in the ``CreateRecord`` routine. In particular the array
has a dimension of ``simulated_time/resolution * n_nodes+1``, where the
first column shows the time simulated and the other columns shows the value of
the parameter recorded for every node. The number of rows and columns
can also be retreived through the commands ``GetRecordDataRows`` and
``GetRecordDataColumns``. Here follows an example:

::

   rows = nestgpu.GetRecordDataRows(recorder)
   columns = nestgpu.GetRecordDataColumns(recorder)
   print("recorder has {} rows and {} columns".format(rows, columns))

   recorded_data = nestgpu.GetRecordData(record)

   time = [row[0] for row in recorded_data]
   variable = [row[1] for row in recorded_data]


See also
++++++++

EndUserDocs */

class Record
{
public:
  bool data_vect_flag_;
  bool out_file_flag_;
  std::vector< std::vector< float > > data_vect_;
  std::vector< BaseNeuron* > neuron_vect_;
  std::string file_name_;
  std::vector< std::string > var_name_vect_;
  std::vector< int > i_neuron_vect_;
  std::vector< int > port_vect_;
  std::vector< float* > var_pt_vect_;
  FILE* fp_;

  Record( std::vector< BaseNeuron* > neur_vect,
    std::string file_name,
    std::vector< std::string > var_name_vect,
    std::vector< int > i_neur_vect,
    std::vector< int > port_vect );

  int OpenFile();

  int CloseFile();

  int WriteRecord( float t, long long time_idx );
};

class Multimeter
{
public:
  std::vector< Record > record_vect_;

  int CreateRecord( std::vector< BaseNeuron* > neur_vect,
    std::string file_name,
    std::vector< std::string > var_name_vect,
    std::vector< int > i_neur_vect,
    std::vector< int > port_vect );
  int OpenFiles();

  int CloseFiles();

  int WriteRecords( float t, long long time_idx );

  std::vector< std::vector< float > >* GetRecordData( int i_record );
};

#endif
