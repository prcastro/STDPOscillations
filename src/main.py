from brian import *
from patterns import *
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

def masquelier(simTime=0.5* second, N=2000, psp=1.4*mV, tau=20*msecond, Vt=-54*mV, Vr=-60*mV, El=-60*mV, R=(10**4)*ohm, oscilFreq=8):
    '''This file executes the simulations (given the parameters),
    of Masquelier's model for learning and saves the results in
    appropriate files.'''

    # Default timestep and number of steps
    dt = defaultclock.dt
    total_steps = float(simTime/dt)
    Nconst_curr=20
    #Size of noise and of drives
    sigma = 0.015*(Vt-Vr)
    Ithr  = ((Vt - El)/R)

    # Neural Model
    neuronEquations =  Equations('''
    dV/dt = -(V-El-R*(I+cI))/tau + sigma*xi/(tau**0.5): volt
    I  : amp
    cI : amp
    ''')

    # Create neuron groups and set initial conditions
    inputLayer = NeuronGroup(N=N, model=neuronEquations, threshold=Vt, reset=Vr)
    inputLayer.V = Vr + rand(N)*(Vt - Vr) # Initial voltage

    # Stablish oscillatory drive
    oscilAmp = 0.15*Ithr
    inputLayer.I = TimedArray((oscilAmp/2)*sin(2*pi*dt*oscilFreq*arange(total_steps) - pi/2))

    #Setting constant drives between .95 and 1.07 * Ithr
    print('Making patterns')
    inputLayer = patterns(inputLayer, simTime, 500, (.95*Ithr,1.07*Ithr))
    print('Patterns are ready')
    #for i in range(N):
    #    inputLayer[i].cI = ((rand())*0.12 + 0.95)*Ithr
    #neuron_poisson = PoissonGroup(N, rates=40*Hz)
    # Connect groups
    #inputDrive = Connection(neuron_poisson, inputLayer)
    #inputDrive.connect_one_to_one(neuron_poisson, inputLayer, weight=psp)

    # Mesurement devices
    spikes = SpikeMonitor(inputLayer)
    voltimeter = StateMonitor(inputLayer, 'V', record=0)
    amperimeter = StateMonitor(inputLayer, 'I', record=0)

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
    return inputLayer


if __name__ == "__main__":
    inputLayer =masquelier()
