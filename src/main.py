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

def masquelier(simTime=0.5*second, N=2000, psp=0.8*mV, tau=20*msecond, V0=-55*mV, Vt=-50*mV, Vr=-60*mV, El=-51*mV):
    '''This file executes the simulations (given the parameters),
    of Masquelier's model for learning and saves the results in
    appropriate files.'''
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
    raster_voltage = figure(1)
    subplot(2,1,1)
    raster_plot(spikes)
    subplot(2,1,2)
    plot(voltimeter.times/ms, voltimeter[0]/mV)
    xlabel('Time (in ms)')
    ylabel('Membrane potential (in mV)')
    title('Membrane potential for neuron 0')
    raster_voltage.show()

if __name__ == "__main__":
    masquelier()
