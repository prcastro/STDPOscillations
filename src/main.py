from brian import *
# import numpy as np

# TODO
# [*] record 2000 neurons spiking with constant drive
# [*] sinusoidal drive
# [ ] stimulus
# [ ] synapse input layer -> output layer
# [ ] Record voltage of output layer
# [ ] STDP
# [ ] Measure learning
# [ ] More realistic model of CA1
# [ ] Parameter search
# [ ] Introduce more patterns

def masquelier(simTime=0.5*second, N=2000, psp=1.4*mV, tau=20*msecond, Vt=-54*mV, Vr=-62*mV, El=-60*mV, R=(10**4)*ohm, oscilFreq=8, constCurrent=40*namp):
    '''This file executes the simulations (given the parameters),
    of Masquelier's model for learning and saves the results in
    appropriate files.'''

    # Default timestep and number of steps
    dt = defaultclock.dt
    total_steps = float(simTime/dt)

    # Neural Model
    neuronEquations =  Equations('''
    dV/dt = -(V-El-R*I)/tau : volt
    I : amp
    ''')

    # Create neuron groups and set initial conditions
    inputSilent = NeuronGroup(N=N, model=neuronEquations, threshold=Vt, reset=Vr)
    inputSilent.V = Vr + rand(N)*(Vt - Vr) # Initial voltage

    # Stablish drives
    oscilAmp = 0.15*((Vt - El)/R)
    inputSilent.I = TimedArray((oscilAmp/2)*sin(2*pi*dt*oscilFreq*arange(total_steps)) + constCurrent)
    neuron_poisson = PoissonGroup(N, rates=40*Hz)

    # Connect groups
    inputDrive = Connection(neuron_poisson, inputSilent)
    inputDrive.connect_one_to_one(neuron_poisson, inputSilent, weight=psp)

    # Mesurement devices
    spikes = SpikeMonitor(inputSilent)
    voltimeter = StateMonitor(inputSilent, 'V', record=0)
    amperimeter = StateMonitor(inputSilent, 'I', record=0)

    # Run the simulation
    run(simTime)

    # Plot raster + voltage of neuron 0
    raster_voltage = figure(1)
    subplot(2,1,1)
    raster_plot(spikes)
    subplot(2,1,2)
    plot(voltimeter.times/ms, voltimeter[0]/mV)
    xlabel('Time (in ms)')
    ylabel('Membrane potential (in mV)')
    title('Membrane potential for neuron 0')
    raster_voltage.show()

    # Plot current at neuron 0 (for debugging purposes, delete after pattern is included)
    current = figure(2)
    plot(amperimeter.times/ms, amperimeter[0]/namp)
    xlabel('Time (in ms)')
    ylabel('Current (in nA)')
    title('Current drive for neuron 0')
    current.show()

if __name__ == "__main__":
    masquelier()
