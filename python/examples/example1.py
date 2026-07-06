"""
NEST GPU Example 1: Basic Single Neuron Simulation
===================================================

This example demonstrates the fundamental workflow for simulating a single
neuron in NEST GPU, from creation to data analysis.

Overview:
---------
This script shows how to:
1. Create a single neuron with a specific model
2. Set neuron parameters
3. Record membrane potential during simulation
4. Run the simulation
5. Extract and visualize the recorded data

Neuron Model:
------------
aeif_cond_beta: Adaptive Exponential Integrate-and-Fire neuron with
conductance-based synapses and beta-function postsynaptic currents.

This neuron model exhibits:
- Adaptive spike frequency
- Exponential approach to threshold
- Conductance-based synaptic inputs
- Beta-function PSC (postsynaptic current) shape

Simulation Details:
-------------------
- Neuron: aeif_cond_beta (1 neuron)
- Current: I_e = 1000.0 pA (constant external current)
- Recording: Membrane potential (V_m)
- Duration: 1000 ms (default)
- Resolution: 0.1 ms (default)

Expected Behavior:
-----------------
With 1000 pA external current, the neuron should fire regular action
potentials. The membrane potential will show:
- Resting potential near -70 mV
- Depolarization when current applied
- Action potentials when threshold reached
- After-hyperpolarization following spikes

Usage:
------
Run this script directly with Python:
    python example1.py

The script will display a plot of membrane potential over time and
wait for user input before closing.

Requirements:
-------------
- NEST GPU library properly installed
- matplotlib for plotting
- CUDA-compatible GPU

Performance:
------------
- Very fast for single neuron simulation
- Minimal GPU memory usage
- Excellent for testing and debugging

Author: NEST Initiative
License: GPL-2.0
"""

import nestgpu as ngpu

# Create a single aeif_cond_beta neuron
# This neuron model combines:
# - Adaptive exponential integrate-and-fire dynamics
# - Conductance-based synaptic inputs
# - Beta-function shaped postsynaptic currents
neuron = ngpu.Create("aeif_cond_beta")

# Set external current to 1000.0 pA
# This constant current will drive the neuron to fire regular action potentials
# I_e is the external current injected into the soma
ngpu.SetStatus(neuron, {"I_e":1000.0})

# Create a recorder to monitor membrane potential
# Parameters:
# - "" : Empty label (no file output)
# - ["V_m"]: Record membrane potential variable
# - [neuron[0]]: Record from the single neuron
# - [0]: Record from port 0 (default port)
record = ngpu.CreateRecord("", ["V_m"], [neuron[0]], [0])

# Run the simulation for default duration (1000 ms)
# During simulation, the neuron will:
# 1. Receive external current I_e
# 2. Integrate membrane potential
# 3. Generate spikes when threshold reached
# 4. Record V_m at each timestep
ngpu.Simulate()

# Retrieve recorded data
# Returns list of [time, value] pairs
data_list = ngpu.GetRecordData(record)

# Extract time and membrane potential for plotting
t = [row[0] for row in data_list]
V_m = [row[1] for row in data_list]

# Visualization using matplotlib
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(t, V_m)

# Add labels and title for better readability
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Single Neuron Simulation - aeif_cond_beta")

# Display the plot
plt.draw()
plt.pause(1)

# Wait for user input before closing (allows viewing the plot)
ngpu.waitenter("<Hit Enter To Close>")
plt.close()
