/*
 *  syn_model.h
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

#ifndef SYNMODEL_H
#define SYNMODEL_H

#include "stdp.h"
#include <string>
#include <vector>

#define MAX_SYN_DT 16384

extern __device__ int* SynGroupTypeMap;
extern __device__ float** SynGroupParamMap;

__device__ void TestSynModelUpdate( float* w, float Dt, float* param );
__device__ void SynapseUpdate( int syn_group, float* w, float Dt, int i_conn = 0 );
#ifdef HAVE_SYN_STATE_VARS
__device__ void SynapsePreTraceUpdate( int syn_group, int i_conn );
__device__ void SynapsePostTraceUpdate( int syn_group, int i_conn );
#endif

enum SynModels
{
  i_null_syn_model = 0,
  i_test_syn_model,
  i_stdp_model,
  i_stdp_synapse_model,
  N_SYN_MODELS
};

const std::string syn_model_name[ N_SYN_MODELS ] = { "", "test_syn_model", "stdp", "stdp_synapse" };

class SynModel
{
protected:
  int type_;
  int n_param_;
  const std::string* param_name_;
  float* d_param_arr_;

#ifdef HAVE_SYN_STATE_VARS
  // State vars
  int n_state_vars_;
  const std::string* state_name_;
#endif

public:
  virtual int
  Init()
  {
    return 0;
  }
  int GetNParam();
  std::vector< std::string > GetParamNames();
  bool IsParam( std::string param_name );
  int GetParamIdx( std::string param_name );
  virtual float GetParam( std::string param_name );
  virtual int SetParam( std::string param_name, float val );

#ifdef HAVE_SYN_STATE_VARS
  // State vars
  int GetNState();
  std::vector< std::string > GetStateNames();
  bool IsState( std::string param_name );
  int GetStateIdx( std::string param_name );
#endif

  friend class NESTGPU;
};

class STDP : public SynModel
{
  int _Init();

public:
  STDP()
  {
    _Init();
  }

  int
  Init()
  {
    return _Init();
  }
};

#endif
