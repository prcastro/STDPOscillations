from brian import *
import numpy as np

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

def masquelier(simTime=0.5*second, N=2000, psp=1.4*mV, tau=20*msecond, Vt=-54*mV, Vr=-62*mV, El=-60*mV, oscilFreq = 8, oscilAmp = 0.004):
    '''This file executes the simulations (given the parameters),
    of Masquelier's model for learning and saves the results in
    appropriate files.'''

    # Default timestep and number of steps
    dt = defaultclock.dt
    total_steps = float(simTime/dt)

    # Neural Model
    neuronEquations =  Equations('''
    dV/dt = -(V-El-I)/tau : volt
    I : volt
    ''')

    # Create neuron groups and set initial conditions
    inputSilent = NeuronGroup(N=N, model=neuronEquations, threshold=Vt, reset=Vr)
    inputSilent.V = Vr + rand(N)*(Vt - Vr)

    # Stablish drives
    inputSilent.I = TimedArray(oscilAmp*cos(2*pi*dt*oscilFreq*arange(total_steps)))
    neuron_poisson = PoissonGroup(N, rates=20*Hz)

    # Connect groups
    inputDrive = Connection(neuron_poisson, inputSilent)
    inputDrive.connect_one_to_one(neuron_poisson, inputSilent, weight=psp)

    # Mesurement devices
    spikes = SpikeMonitor(inputSilent)
    voltimeter = StateMonitor(inputSilent, 'V', record=0)

    # Run the simulation
    run(simTime)

    # Ploting
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
