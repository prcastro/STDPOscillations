from brian import *
import numpy as np

# TODO
# 1) record 2000 neurons spiking with constant drive
# 2) sinusoidal drive
# 3) stimulus
# 4) synapse input layer -> output layer
# 5) Record voltage of output layer
# 6) STDP
# 7) Measure learning
# 8) More realistic model of CA1
# 9) Parameter search
# 10) Introduce more patterns

def masquelier():
    '''This file executes the simulations (given the parameters),
    of Masquelier's model for learning and saves the results in
    appropriate files.'''
    # Model parameters (some of these will end up as arguments of this function)
    simTime =    0.5 * second  # Duration of the simulation
    N       = 2000             # Number of input neurons
    psp     =    0.8 * mV      # Post-Synaptic Potential
    tau     =   20   * msecond # membrane time constant
    V0      =  -55   * mV      # Initial voltage
    Vt      =  -50   * mV      # spike threshold
    Vr      =  -60   * mV      # reset value
    El      =  -51   * mV      # resting potential (same as the reset)
    # Neural Model
    neuronEquations =  Equations('''
    dV/dt = -(V-El)/tau : volt
    ''')
    # Create neuron groups and set initial conditions
    inputSilent = NeuronGroup(N=N, model=neuronEquations, threshold=Vt, reset=Vr)
    inputSilent.V = V0*ones(N)
    # Stablish drive
    poisson = PoissonGroup(N,rates=20*Hz)
    # Connect groups
    inputDrive = Connection(poisson, inputSilent)
    inputDrive.connect_one_to_one(poisson, inputSilent, weight=psp)
    # Mesurement devices
    spikes = SpikeMonitor(inputSilent)
    voltimeter = StateMonitor(inputSilent, 'V', record=0)
    # Run the simulation
    run(simTime)
    # Measurements
    # Return/plot/save the data
    raster = figure(1)
    raster_plot(spikes)
    raster.show()
    voltage = figure(2)
    plot(voltimeter.times/ms, voltimeter[0]/mV)
    xlabel('Time (in ms)')
    ylabel('Membrane potential (in mV)')
    title('Membrane potential for neuron 0')
    voltage.show()


if __name__ == "__main__":
    masquelier()
