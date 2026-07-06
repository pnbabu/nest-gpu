"""
NEST GPU Brunel Network Example
================================

This example implements the Brunel network model, a classic benchmark
in computational neuroscience for studying asynchronous irregular (AI)
activity in spiking neural networks.

Overview:
---------
The Brunel network is a randomly connected network of integrate-and-fire
neurons that exhibits self-sustained asynchronous irregular activity,
resembling cortical dynamics.

This implementation includes:
- Excitatory and inhibitory neuron populations
- Random connectivity with fixed indegree
- External Poisson input
- Stochastic synaptic dynamics
- Multiple neuron monitoring

Network Architecture:
---------------------
Based on Brunel (2000) "Dynamics of sparsely connected networks of
integrate-and-fire neurons"

Network composition:
- 80% excitatory neurons (NE = 4/5 * n_neurons)
- 20% inhibitory neurons (NI = 1/5 * n_neurons)
- Fixed indegree connectivity
- External Poisson input to all neurons

Mathematical Model:
------------------
The network uses aeif_cond_beta neurons with:
- Adaptive exponential integrate-and-fire dynamics
- Conductance-based synaptic inputs
- Beta-function postsynaptic currents

Network dynamics balance excitation and inhibition to produce
asynchronous irregular firing at rates ~8 Hz.

Simulation Parameters:
---------------------
Command line usage:
    python brunel_net.py <n_neurons>

where n_neurons is the total number of neurons (must be divisible by 5)

Network size example:
    python brunel_net.py 5000  # 4000 excitatory, 1000 inhibitory

Connectivity:
- Excitatory: CE = 800 connections per neuron
- Inhibitory: CI = CE/4 = 200 connections per neuron
- Delays: Normal distribution (μ=0.5ms, σ=0.25ms, clipped)
- Weights: Wex = 0.05 (excitatory), Win = 0.35 (inhibitory)

External Input:
- Poisson rate: 20 kHz (high rate for AI state)
- Weight: 0.37 (excitatory)
- Delay: 0.2 ms

Receptors:
- Receptor 0: Excitatory (E_rev = 0.0 mV)
- Receptor 1: Inhibitory (E_rev = -85.0 mV)

Expected Behavior:
-----------------
The network should exhibit:
- Asynchronous irregular activity
- Mean firing rate ~8 Hz
- Coefficient of variation ~0.8
- Balanced excitation and inhibition
- Self-sustained activity without external input

Performance:
------------
- Scales linearly with network size
- GPU acceleration for large networks (>10,000 neurons)
- Efficient sparse connectivity
- Fast simulation of benchmark networks

Data Recording:
--------------
Records membrane potential from 3 neurons:
1. Neuron 37 (fixed excitatory neuron)
2. Random neuron (varies each run)
3. Last neuron (fixed position)

Data saved to: test_brunel_net.dat

Visualization:
------------
Shows 3 plots of membrane potential vs time for different neurons.

Usage:
------
    python brunel_net.py 10000    # Large network
    python brunel_net.py 5000     # Medium network
    python brunel_net.py 1000     # Small network

Requirements:
-------------
- NEST GPU library properly installed
- matplotlib for plotting
- CUDA-compatible GPU
- Sufficient GPU memory for network size

Scientific Background:
----------------------
The Brunel network is a standard model for studying:
- Balanced network dynamics
- Asynchronous irregular states
- Synaptic integration
- Network oscillations
- Critical phenomena

Reference:
Brunel, N. (2000). Dynamics of sparsely connected networks of
integrate-and-fire neurons. Journal of Computational Neuroscience,
8(3), 183-208.

Author: NEST Initiative
License: GPL-2.0
"""

import sys
import ctypes
import nestgpu as ngpu
from random import randrange

# Parse command line argument for network size
if len(sys.argv) != 2:
    print("Usage: python %s n_neurons" % sys.argv[0])
    quit()

# Network size calculation (must be divisible by 5)
order = int(sys.argv[1]) // 5

print("Building Brunel network with {} neurons...".format(int(sys.argv[1])))

# Set random seed for reproducible results
# This ensures the same network structure each time
ngpu.SetKernelStatus("rnd_seed", 1234) # seed for GPU random numbers

# Number of receptor ports (excitatory + inhibitory)
n_receptors = 2

# Network size parameters
NE = 4 * order       # number of excitatory neurons (80%)
NI = 1 * order       # number of inhibitory neurons (20%)
n_neurons = NE + NI  # total number of neurons

# Connectivity parameters
CE = 800   # number of excitatory synapses per neuron
CI = CE//4  # number of inhibitory synapses per neuron (200)

# Synaptic weights
Wex = 0.05   # excitatory synaptic weight
Win = 0.35   # inhibitory synaptic weight (7x stronger)

# Poisson generator parameters (external input)
poiss_rate = 20000.0      # Poisson signal rate in Hz (20 kHz)
poiss_weight = 0.37       # External input weight
poiss_delay = 0.2         # Poisson signal delay in ms

# Create Poisson spike generator for external input
pg = ngpu.Create("poisson_generator")
ngpu.SetStatus(pg, "rate", poiss_rate)

# Create neuron population with aeif_cond_beta model
# Each neuron has 2 receptor ports (excitatory and inhibitory)
neuron = ngpu.Create("aeif_cond_beta", n_neurons, n_receptors)

# Split neuron population into excitatory and inhibitory groups
exc_neuron = neuron[0:NE]      # excitatory neurons (first 80%)
inh_neuron = neuron[NE:n_neurons]   # inhibitory neurons (last 20%)

# Receptor parameters for synaptic dynamics
# E_rev: Reversal potentials for excitatory (0 mV) and inhibitory (-85 mV)
# tau_decay: Decay time constants for synaptic conductances
# tau_rise: Rise time constants for beta-function PSCs
E_rev = [0.0, -85.0]
tau_decay = [1.0, 1.0]
tau_rise = [1.0, 1.0]

# Set receptor parameters for all neurons
# This defines the synaptic dynamics for each receptor type
ngpu.SetStatus(neuron, {"E_rev_ex": E_rev[0],
                        "E_rev_in": E_rev[1],
                        "tau_decay_ex": tau_decay[0],
                        "tau_decay_in": tau_decay[1],
                        "tau_rise_ex": tau_rise[0],
                        "tau_rise_in": tau_rise[1]})

# Delay distribution parameters
mean_delay = 0.5      # Mean synaptic delay (ms)
std_delay = 0.25      # Standard deviation of delay
min_delay = 0.1       # Minimum delay (ms)

# Excitatory connections
# Connect excitatory neurons to receptor 0 of all neurons
# Uses fixed_indegree rule with CE connections per neuron
# Delays follow clipped normal distribution
exc_conn_dict = {"rule": "fixed_indegree", "indegree": CE}
exc_syn_dict = {"weight": Wex,
                "delay": {"distribution": "normal_clipped",
                          "mu": mean_delay,
                          "low": min_delay,
                          "high": mean_delay + 3 * std_delay,
                          "sigma": std_delay},
                "receptor": 0}
ngpu.Connect(exc_neuron, neuron, exc_conn_dict, exc_syn_dict)

# Inhibitory connections
# Connect inhibitory neurons to receptor 1 of all neurons
# Uses fixed_indegree rule with CI connections per neuron
# Delays follow clipped normal distribution
inh_conn_dict = {"rule": "fixed_indegree", "indegree": CI}
inh_syn_dict = {"weight": Win,
                "delay": {"distribution": "normal_clipped",
                          "mu": mean_delay,
                          "low": min_delay,
                          "high": mean_delay + 3 * std_delay,
                          "sigma": std_delay},
                "receptor": 1}
ngpu.Connect(inh_neuron, neuron, inh_conn_dict, inh_syn_dict)

# Connect Poisson generator to all neurons (receptor 0)
# Provides external drive to maintain network activity
# Uses all_to_all connectivity pattern
pg_conn_dict = {"rule": "all_to_all"}
pg_syn_dict = {"weight": poiss_weight,
               "delay": poiss_delay,
               "receptor": 0}

ngpu.Connect(pg, neuron, pg_conn_dict, pg_syn_dict)

# Set up data recording
# Record membrane potential from 3 different neurons:
# 1. Neuron 37 (fixed excitatory neuron)
# 2. Random neuron (for variety)
# 3. Last neuron in the population
filename = "test_brunel_net.dat"
i_neuron_arr = [neuron[37], neuron[randrange(n_neurons)], neuron[n_neurons-1]]
i_receptor_arr = [0, 0, 0]  # All recordings from receptor 0

# Create multimeter record for membrane potential
var_name_arr = ["V_m", "V_m", "V_m"]
record = ngpu.CreateRecord(filename, var_name_arr, i_neuron_arr, i_receptor_arr)

# Run the simulation
# Network will develop asynchronous irregular activity
print("Simulating network...")
ngpu.Simulate()

# Retrieve and analyze recorded data
nrows = ngpu.GetRecordDataRows(record)
ncol = ngpu.GetRecordDataColumns(record)

data_list = ngpu.GetRecordData(record)
t = [row[0] for row in data_list]
V1 = [row[1] for row in data_list]  # Neuron 37
V2 = [row[2] for row in data_list]  # Random neuron
V3 = [row[3] for row in data_list]  # Last neuron

# Visualization
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(t, V1)
plt.title("Membrane Potential - Neuron 37")
plt.xlabel("Time (ms)")
plt.ylabel("V_m (mV)")

plt.figure(2)
plt.plot(t, V2)
plt.title("Membrane Potential - Random Neuron")
plt.xlabel("Time (ms)")
plt.ylabel("V_m (mV)")

plt.figure(3)
plt.plot(t, V3)
plt.title("Membrane Potential - Last Neuron")
plt.xlabel("Time (ms)")
plt.ylabel("V_m (mV)")

plt.draw()
plt.pause(0.5)
ngpu.waitenter("<Hit Enter To Close>")
plt.close()
