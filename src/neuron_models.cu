/*
 *  neuron_models.cu
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
 * @file neuron_models.cu
 * @brief Neuron model factory and management for NEST GPU
 *
 * This file implements the factory pattern for creating and managing
 * different neuron models in NEST GPU. It provides a unified interface
 * for instantiating various neuron types and managing their lifecycle.
 *
 * Architecture Overview:
 * ---------------------
 * The neuron model system uses a factory pattern to create neuron
 * instances based on string names. Each neuron model is implemented
 * as a separate class inheriting from BaseNeuron.
 *
 * Available Neuron Models:
 * ------------------------
 * Integrate-and-Fire Models:
 * - iaf_psc_exp: Leaky integrate-and-fire with exponential PSC
 * - iaf_psc_alpha: Leaky integrate-and-fire with alpha PSC
 * - iaf_psc_exp_g: Conductance-based IAF with exponential PSC
 * - iaf_psc_exp_hc: IAF with homeostatic control
 *
 * Adaptive Exponential Models:
 * - aeif_cond_alpha: Adaptive exponential with conductance-based alpha PSC
 * - aeif_cond_beta: Adaptive exponential with conductance-based beta PSC
 * - aeif_psc_exp: Adaptive exponential with current-based exponential PSC
 * - aeif_psc_alpha: Adaptive exponential with current-based alpha PSC
 * - aeif_psc_delta: Adaptive exponential with delta PSC
 *
 * Multisynapse Models:
 * - aeif_cond_alpha_multisynapse: Multi-receptor adaptive exponential
 * - aeif_cond_beta_multisynapse: Multi-receptor adaptive beta
 * - aeif_psc_exp_multisynapse: Multi-receptor current-based
 * - aeif_psc_alpha_multisynapse: Multi-receptor alpha current
 *
 * Izhikevich Models:
 * - izhikevich: Simple spiking model
 * - izhikevich_psc_exp: Izhikevich with exponential PSC
 * - izhikevich_cond_beta: Izhikevich with conductance-based beta
 * - izhikevich_psc_exp_2s: Two-compartment Izhikevich
 * - izhikevich_psc_exp_5s: Five-compartment Izhikevich
 *
 * Special Models:
 * - parrot_neuron: Repeat incoming spikes
 * - ext_neuron: External neuron for hybrid simulations
 * - poiss_gen: Poisson spike generator
 * - spike_generator: Custom spike train generator
 * - spike_detector: Spike recording device
 *
 * User-Defined Models:
 * - user_m1: Template for user-defined model type 1
 * - user_m2: Template for user-defined model type 2
 * - Various user model variants with different synaptic dynamics
 *
 * NESTML-Generated Models:
 * - ca_adex_alt_nestml: Calcium-based adaptive exponential
 * - aeif_cond_alpha_neuron_nestml: NESTML-generated adaptive model
 * - iaf_psc_exp_neuron_nestml: NESTML-generated IAF model
 *
 * Factory Pattern:
 * ----------------
 * The _Create() function implements the factory:
 * - Takes model name as string input
 * - Creates appropriate neuron instance
 * - Sets default number of ports based on model
 * - Returns NodeSeq handle for accessing neurons
 *
 * Model Registration:
 * ------------------
 * Each model is registered with:
 * - Unique name string
 * - Default number of ports
 * - Parameter definitions
 * - Variable definitions
 * - Update and calibration functions
 *
 * Port Configuration:
 * ------------------
 * Different models have different port requirements:
 * - Single receptor models: 1 port (e.g., iaf_psc_exp_g)
 * - Excitatory/inhibitory: 2 ports (e.g., iaf_psc_exp)
 * - Multisynapse: Variable ports (e.g., aeif_cond_alpha_multisynapse)
 *
 * Memory Management:
 * -----------------
 * - Neuron instances stored in node_vect_ vector
 * - Each model manages its own GPU memory
 * - Automatic cleanup on destruction
 * - Efficient memory pooling for same-type neurons
 *
 * Integration Points:
 * ------------------
 * - NESTGPU::Create(): Public interface for model creation
 * - BaseNeuron: Common interface for all models
 * - Individual model headers: Specific implementations
 * - CUDA kernels: GPU-side update functions
 *
 * Usage Pattern:
 * --------------
 * 1. Call Create(model_name, n_neurons, n_ports)
 * 2. System creates appropriate neuron instance
 * 3. Set neuron parameters
 * 4. Connect neurons with synapses
 * 5. Run simulation
 *
 * Error Handling:
 * --------------
 * - Invalid model names throw exceptions
 * - Zero neuron count validation
 * - Negative port count validation
 * - Post-calibration creation prevention
 *
 * Performance Considerations:
 * ---------------------------
 * - Model creation is CPU-side operation
 * - Batch creation of neurons efficient
 * - Memory allocated once per neuron group
 * - Minimal overhead during simulation
 *
 * Thread Safety:
 * --------------
 * - Model creation is not thread-safe
 * - Multiple neuron groups can be created sequentially
 * - Simulation updates are thread-safe on GPU
 *
 * Extension:
 * ---------
 * Adding new neuron models:
 * 1. Create model class inheriting from BaseNeuron
 * 2. Add model to neuron_model_names_ array
 * 3. Update model enum values
 * 4. Implement required virtual methods
 * 5. Add Create() case for new model
 *
 * Scientific Background:
 * ----------------------
 * Models are based on published research:
 * - IAF: Classic integrate-and-fire neuron
 * - AdEx: Adaptive exponential (Brette & Gerstner, 2005)
 * - Izhikevich: Simple model (Izhikevich, 2003)
 * - PSC variations: Different synaptic dynamics
 *
 * @see neuron_models.h Model definitions and enums
 * @see base_neuron.h Base neuron interface
 * @see nestgpu.h Main simulation engine
 */

#include <config.h>
#include <iostream>
#include <string>

#include "aeif_cond_alpha.h"
#include "aeif_cond_alpha_multisynapse.h"
#include "aeif_cond_beta.h"
#include "aeif_cond_beta_multisynapse.h"
#include "aeif_psc_alpha.h"
#include "aeif_psc_alpha_multisynapse.h"
#include "aeif_psc_delta.h"
#include "aeif_psc_exp.h"
#include "aeif_psc_exp_multisynapse.h"
#include "cuda_error.h"
#include "ext_neuron.h"
#include "getRealTime.h"
#include "iaf_psc_alpha.h"
#include "iaf_psc_exp.h"
#include "iaf_psc_exp_g.h"
#include "iaf_psc_exp_hc.h"
#include "izhikevich.h"
#include "izhikevich_cond_beta.h"
#include "izhikevich_psc_exp.h"
#include "izhikevich_psc_exp_2s.h"
#include "izhikevich_psc_exp_5s.h"
#include "nestgpu.h"
#include "neuron_models.h"
#include "ngpu_exception.h"
#include "parrot_neuron.h"
#include "poiss_gen.h"
#include "spike_detector.h"
#include "spike_generator.h"
#include "user_m1.h"
#include "user_m2.h"
// <<BEGIN_NESTML_GENERATED>>
#include "ca_adex_alt_nestml.h"
// <<END_NESTML_GENERATED>>
NodeSeq
NESTGPU::_Create( std::string model_name, uint n_nodes /*=1*/, int n_ports /*=1*/ )
{
  if ( !create_flag_ )
  {
    create_flag_ = true;
    start_real_time_ = getRealTime();
  }
  CheckUncalibrated( "Nodes cannot be created after calibration" );
  if ( n_nodes <= 0 )
  {
    throw ngpu_exception( "Number of nodes must be greater than zero." );
  }
  else if ( n_ports < 0 )
  {
    throw ngpu_exception( "Number of ports must be >= zero." );
  }
  if ( model_name == neuron_model_name[ i_iaf_psc_exp_g_model ] )
  {
    n_ports = 1;
    iaf_psc_exp_g* iaf_psc_exp_g_group = new iaf_psc_exp_g;
    node_vect_.push_back( iaf_psc_exp_g_group );
  }
  else if ( model_name == neuron_model_name[ i_iaf_psc_exp_hc_model ] )
  {
    n_ports = 1;
    iaf_psc_exp_hc* iaf_psc_exp_hc_group = new iaf_psc_exp_hc;
    node_vect_.push_back( iaf_psc_exp_hc_group );
  }
  else if ( model_name == neuron_model_name[ i_iaf_psc_exp_model ] )
  {
    n_ports = 2;
    iaf_psc_exp* iaf_psc_exp_group = new iaf_psc_exp;
    node_vect_.push_back( iaf_psc_exp_group );
  }
  else if ( model_name == neuron_model_name[ i_iaf_psc_alpha_model ] )
  {
    n_ports = 2;
    iaf_psc_alpha* iaf_psc_alpha_group = new iaf_psc_alpha;
    node_vect_.push_back( iaf_psc_alpha_group );
  }
  else if ( model_name == neuron_model_name[ i_ext_neuron_model ] )
  {
    ext_neuron* ext_neuron_group = new ext_neuron;
    node_vect_.push_back( ext_neuron_group );
  }
  else if ( model_name == neuron_model_name[ i_aeif_cond_alpha_model ] )
  {
    n_ports = 2;
    aeif_cond_alpha* aeif_cond_alpha_group = new aeif_cond_alpha;
    node_vect_.push_back( aeif_cond_alpha_group );
  }
  else if ( model_name == neuron_model_name[ i_aeif_cond_beta_model ] )
  {
    n_ports = 2;
    aeif_cond_beta* aeif_cond_beta_group = new aeif_cond_beta;
    node_vect_.push_back( aeif_cond_beta_group );
  }
  else if ( model_name == neuron_model_name[ i_aeif_psc_alpha_model ] )
  {
    n_ports = 2;
    aeif_psc_alpha* aeif_psc_alpha_group = new aeif_psc_alpha;
    node_vect_.push_back( aeif_psc_alpha_group );
  }
  else if ( model_name == neuron_model_name[ i_aeif_psc_delta_model ] )
  {
    n_ports = 1;
    aeif_psc_delta* aeif_psc_delta_group = new aeif_psc_delta;
    node_vect_.push_back( aeif_psc_delta_group );
  }
  else if ( model_name == neuron_model_name[ i_aeif_psc_exp_model ] )
  {
    n_ports = 2;
    aeif_psc_exp* aeif_psc_exp_group = new aeif_psc_exp;
    node_vect_.push_back( aeif_psc_exp_group );
  }
  else if ( model_name == neuron_model_name[ i_aeif_cond_beta_multisynapse_model ] )
  {
    aeif_cond_beta_multisynapse* aeif_cond_beta_multisynapse_group = new aeif_cond_beta_multisynapse;
    node_vect_.push_back( aeif_cond_beta_multisynapse_group );
  }
  else if ( model_name == neuron_model_name[ i_aeif_cond_alpha_multisynapse_model ] )
  {
    aeif_cond_alpha_multisynapse* aeif_cond_alpha_multisynapse_group = new aeif_cond_alpha_multisynapse;
    node_vect_.push_back( aeif_cond_alpha_multisynapse_group );
  }
  else if ( model_name == neuron_model_name[ i_aeif_psc_exp_multisynapse_model ] )
  {
    aeif_psc_exp_multisynapse* aeif_psc_exp_multisynapse_group = new aeif_psc_exp_multisynapse;
    node_vect_.push_back( aeif_psc_exp_multisynapse_group );
  }
  else if ( model_name == neuron_model_name[ i_aeif_psc_alpha_multisynapse_model ] )
  {
    aeif_psc_alpha_multisynapse* aeif_psc_alpha_multisynapse_group = new aeif_psc_alpha_multisynapse;
    node_vect_.push_back( aeif_psc_alpha_multisynapse_group );
  }
  else if ( model_name == neuron_model_name[ i_user_m1_model ] )
  {
    user_m1* user_m1_group = new user_m1;
    node_vect_.push_back( user_m1_group );
  }
  else if ( model_name == neuron_model_name[ i_user_m2_model ] )
  {
    user_m2* user_m2_group = new user_m2;
    node_vect_.push_back( user_m2_group );
  }
  else if ( model_name == neuron_model_name[ i_poisson_generator_model ] )
  {
    n_ports = 0;
    poiss_gen* poiss_gen_group = new poiss_gen;
    node_vect_.push_back( poiss_gen_group );
  }
  else if ( model_name == neuron_model_name[ i_spike_generator_model ] )
  {
    n_ports = 0;
    spike_generator* spike_generator_group = new spike_generator;
    node_vect_.push_back( spike_generator_group );
  }
  else if ( model_name == neuron_model_name[ i_parrot_neuron_model ] )
  {
    n_ports = 2;
    parrot_neuron* parrot_neuron_group = new parrot_neuron;
    node_vect_.push_back( parrot_neuron_group );
  }
  else if ( model_name == neuron_model_name[ i_spike_detector_model ] )
  {
    n_ports = 1;
    spike_detector* spike_detector_group = new spike_detector;
    node_vect_.push_back( spike_detector_group );
  }
  else if ( model_name == neuron_model_name[ i_izhikevich_model ] )
  {
    izhikevich* izhikevich_group = new izhikevich;
    node_vect_.push_back( izhikevich_group );
  }
  else if ( model_name == neuron_model_name[ i_izhikevich_cond_beta_model ] )
  {
    izhikevich_cond_beta* izhikevich_cond_beta_group = new izhikevich_cond_beta;
    node_vect_.push_back( izhikevich_cond_beta_group );
  }
  else if ( model_name == neuron_model_name[ i_izhikevich_psc_exp_5s_model ] )
  {
    izhikevich_psc_exp_5s* izhikevich_psc_exp_5s_group = new izhikevich_psc_exp_5s;
    node_vect_.push_back( izhikevich_psc_exp_5s_group );
  }
  else if ( model_name == neuron_model_name[ i_izhikevich_psc_exp_2s_model ] )
  {
    izhikevich_psc_exp_2s* izhikevich_psc_exp_2s_group = new izhikevich_psc_exp_2s;
    node_vect_.push_back( izhikevich_psc_exp_2s_group );
  }
  else if ( model_name == neuron_model_name[ i_izhikevich_psc_exp_model ] )
  {
    izhikevich_psc_exp* izhikevich_psc_exp_group = new izhikevich_psc_exp;
    node_vect_.push_back( izhikevich_psc_exp_group );
  }
  // <<BEGIN_NESTML_GENERATED>>
else if (model_name == neuron_model_name[i_ca_adex_alt_nestml_model]) {
    n_ports = 7;
    ca_adex_alt_nestml *ca_adex_alt_nestml_group = new ca_adex_alt_nestml;
    node_vect_.push_back(ca_adex_alt_nestml_group);
 }
// <<END_NESTML_GENERATED>>
  else
  {
    throw ngpu_exception( std::string( "Unknown neuron model name: " ) + model_name );
  }
  return NodeSeq( CreateNodeGroup( n_nodes, n_ports ), n_nodes );
}
NodeSeq
NESTGPU::Create( std::string model_name, uint n_nodes, int n_ports )
{
  for ( int i_host = 0; i_host < n_hosts_; i_host++ )
  {
    n_remote_nodes_[ i_host ] += n_nodes;
  }

  return _Create( model_name, n_nodes, n_ports );
}
