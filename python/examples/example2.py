"""
NEST GPU Example 2: Neuron Driven by Poisson Spike Generator
============================================================

This example demonstrates how to drive a neuron with stochastic input
from a Poisson spike generator, simulating realistic synaptic input.

Overview:
---------
This script shows how to:
1. Create a neuron and a Poisson spike generator
2. Set spike generator rate
3. Connect the generator to the neuron
4. Record and visualize membrane potential dynamics

Network Components:
------------------
- 1 aeif_cond_beta neuron: Adaptive exponential integrate-and-fire neuron
- 1 Poisson spike generator: Generates spikes at random times
- Connection: One-to-one connection with specified weight and delay

Simulation Details:
-------------------
- Poisson rate: 12000 Hz (high frequency for clear response)
- Synaptic weight: 0.05 (excitatory conductance)
- Synaptic delay: 2.0 ms
- Receptor type: 0 (default receptor)
- Duration: 1000 ms (default)
- Resolution: 0.1 ms (default)

Poisson Spike Generator:
-----------------------
Generates spikes stochastically with a constant rate (12000 Hz).
The interspike intervals follow an exponential distribution:
    P(t) = rate * exp(-rate * t)

At 12 kHz, average interspike interval = 1/12000 ≈ 0.083 ms

Expected Behavior:
-----------------
The neuron should show:
- Subthreshold membrane potential fluctuations
- Postsynaptic potentials (PSPs) from each spike
- Occasional spike firing if input is strong enough
- Irregular firing pattern due to stochastic input

The high Poisson rate (12 kHz) ensures many PSPs per second,
creating significant membrane potential fluctuations.

Connection Details:
-------------------
- Rule: one_to_one (paired connection)
- Weight: 0.05 (excitatory, increases membrane potential)
- Delay: 2.0 ms (spike delivery delay)
- Receptor: 0 (default excitatory receptor)

Usage:
------
Run this script directly with Python:
    python example2.py

The script will display a plot of membrane potential over time.

Requirements:
-------------
- NEST GPU library properly installed
- matplotlib for plotting
- CUDA-compatible GPU

Performance:
------------
- Very fast for single neuron with Poisson input
- Minimal GPU memory usage
- Excellent for testing stochastic integration

Author: NEST Initiative
License: GPL-2.0
"""

import nestgpu as ngpu

# Create a single aeif_cond_beta neuron
# This neuron model exhibits adaptive firing patterns and
# realistic synaptic integration
neuron = ngpu.Create("aeif_cond_beta")

# Create a Poisson spike generator
# This device generates spikes stochastically at a specified rate
poiss_gen = ngpu.Create("poisson_generator")

# Set Poisson generator firing rate to 12000 Hz
# High rate ensures many synaptic inputs per second
# Rate parameter determines the average number of spikes per second
ngpu.SetStatus(poiss_gen, "rate", 12000.0)

# Define connection rule (one-to-one pairing)
# This creates a direct connection between the generator and neuron
conn_dict = {"rule": "one_to_one"}

# Define synaptic properties
# - weight: 0.05 increases membrane potential (excitatory)
# - delay: 2.0 ms synaptic transmission delay
# - receptor: 0 uses default excitatory receptor
syn_dict = {"weight": 0.05, "delay": 2.0, "receptor": 0}

# Connect the Poisson generator to the neuron
# Each spike from the generator will cause a postsynaptic potential
# in the neuron after the specified delay
ngpu.Connect(poiss_gen, neuron, conn_dict, syn_dict)

# Create recorder to monitor membrane potential
record = ngpu.CreateRecord("", ["V_m"], [neuron[0]], [0])

# Run the simulation
# During simulation:
# - Poisson generator generates spikes stochastically
# - Spikes are delivered to neuron after 2.0 ms delay
# - Each spike causes a postsynaptic potential
# - Neuron may fire if membrane potential exceeds threshold
ngpu.Simulate()

# Extract recorded data for plotting
data_list = ngpu.GetRecordData(record)
t = [row[0] for row in data_list]
V_m = [row[1] for row in data_list]

# Visualization
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(t, V_m)

# Add informative labels
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Neuron Driven by Poisson Spike Generator (12 kHz)")

plt.draw()
plt.pause(1)
ngpu.waitenter("<Hit Enter To Close>")
plt.close()
