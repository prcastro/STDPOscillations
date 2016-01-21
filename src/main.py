from brian import *
import matplotlib.gridspec as gridspec
# style.use('ggplot')

# TODO
# [*] record 2000 neurons spiking with constant drive
# [*] sinusoidal drive
# [*] stimulus
# [*] synapse input layer -> output layer
# [*] Record voltage of output layer
# [*] STDP
# [ ] Measure learning
# [ ] More realistic model of CA1
# [ ] Parameter search
# [ ] Introduce more patterns

def plotActivations(values, times, pattern_presence):
    N = len(values[0])
    discrete_actv=[[] for i in range(N)]
    presence = [[] for i in range(20)]
    for n in range(N):
        for ti in range(1,len(times)):
            t = times[ti-1]
            while t < times[ti]:
                discrete_actv[n] += [values[ti-1][n]]
                if n in range(20):
                    presence[n] +=[pattern_presence[ti-1]]
                t+=0.001 # In seconds

    matshow(discrete_actv + presence, cmap=plt.cm.gray)
    show()


def activationLevels(N, totalTime, pattern, toPlot=True):
    '''This function return the activation levels matrix with
    the level of activation of each neuron over time. This is returned as
    a TimedArray. The list of the times in each the pattern in present is
    returned as well'''
    mean_dt   = 250e-3
    mean_dpat = 1250e-3

    # Time intervals
    times = []
    t = 0
    while t < totalTime:
        times.append(t)
        t+= np.random.exponential(mean_dt)
    times.append(totalTime)

    # Time of pattern arrivals
    ptimes = []
    t = exponential(mean_dpat)
    while t < totalTime:
        ptimes.append(t)
        t += np.random.exponential(mean_dpat)

    # Put patterns' times in the array, and sort
    times += ptimes
    times  = sort(times)

    # Random activation matrix - the last 21 rows are for pattern
    #  identification (grey when present and white when not)
    activations = rand(len(times), N)
    pattern_presence = ones(len(times))

    # Add patterns to activation matrix, and also the identification at
    #  the bottom
    for (i,t) in enumerate(times):
        if t in ptimes:
            activations[i, (N - len(pattern)):N] = pattern
            pattern_presence[i] = 0.8

    if toPlot:
        plotActivations(activations, times, pattern_presence)

    return TimedArray(activations,times), pattern_presence


def masquelier(simTime=3*second, N=2000, Vt=-54*mV, Vr=-60*mV, El=-70*mV, tau=20*ms, taus=5*ms, taup=16.8*ms, taum=33.7*ms, aplus=0.005, aratio=1.48, R=9*(10**6)*ohm, oscilFreq=8, patt_act=rand(200), patt_range=(1800,2000), toPlot=True):
    '''This file executes the simulations (given the parameters),
    of Masquelier's model for learning and saves the results in
    appropriate files.

    Note: R is nine times larger than in the paper'''

    if len(patt_act) != (patt_range[1] - patt_range[0]):
        raise ValueError("Pattern activation must be consistent with pattern range")

    # Default timestep and number of steps
    dt = defaultclock.dt
    total_steps = float(simTime/dt)

    # Size of noise, drives and weights
    sigma = 0.015*(Vt - Vr)
    Ithr  = (Vt - El)/R
    Imax  = 0.05*namp
    wmax  = 2*(8.6 * pamp/Imax)

    # Model of input neuron. Current comes from oscillatory input (I) and from
    # the activation levels (actValue, like Masquelier et al 2009 Figure 1). This
    # is based on Eq 1 of Masquelier et al 2009.
    inputNeuron =  Equations('''
    dV/dt = -(V-El - R*(I+actValue))/tau + sigma*xi/(tau**0.5): volt
    I: amp
    actValue: amp
    ''')

    outputNeuron =  Equations('''
    dV/dt = -(V-El - R*(s*Imax))/tau + sigma*xi/(tau**0.5): volt
    ds/dt = -s/taus: 1
    ''')

    # Create neuron groups and set initial conditions
    inputLayer    = NeuronGroup(N=N, model=inputNeuron, threshold=Vt, reset=Vr)
    inputLayer.V  = Vr + rand(N)*(Vt - Vr) # Initial voltage
    outputLayer   = NeuronGroup(N=1, model=outputNeuron, threshold=Vt, reset=Vr)
    outputLayer.V = Vr + rand()*(Vt - Vr) # Initial voltage

    # Oscillatory drive on the input layer
    oscilAmp     = 0.15*Ithr
    inputLayer.I = TimedArray((oscilAmp/2)*sin(2*pi*oscilFreq*dt*arange(total_steps) - pi))

    # Get the activation levels' matrix and use as input current
    acts, _ = activationLevels(N, simTime/second, patt_act, toPlot=False)
    inputLayer.actValue = (acts*0.12 + 0.95)*Ithr # Affine mapping between activation and input

    # # Connect the layers
    weights = rand(N, 1)*wmax
    con     = Connection(inputLayer, outputLayer, 's', weight=weights)

    # # STDP synapse
    aminus = -(aplus * aratio)
    stdp = ExponentialSTDP(con, taup, taum, aplus, aminus, wmax=1, interactions='all', update='additive')

    # Mesurement devices
    spikes      = SpikeMonitor(inputLayer[patt_range[0]-50:patt_range[1]-50])
    voltimeter  = StateMonitor(outputLayer, 'V', record=0)
    amperimeter = StateMonitor(inputLayer, 'I', record=0)

    # Run the simulation
    run(simTime, report='text')

    if toPlot:
        # Update weights to the values found on the Connection object
        weights = con.W.todense()

        # Plot raster + voltage of neuron 0
        raster_voltage = figure(1)

        # Set grid
        gs = gridspec.GridSpec(3, 2)
        gs.update(hspace=0.5)

        # Raster plot
        subplot(gs[0,:])
        raster_plot(spikes, markersize=4)
        ylabel('Afferent #')
        xlabel('Time (in ms)', fontsize=10)

        # Plot membrane potential of the output
        subplot(gs[1,:])
        plot(voltimeter.times/second, voltimeter[0]/mV)
        axhline(-54, linestyle=':')
        ylim([-70, -53])
        xlabel('Time (in s)', fontsize=10)
        ylabel('Membrane potential (in mV)', fontsize=10)

        # Weights' histogram
        subplot(gs[2,0])
        hist(weights, 9)
        xlim([0.0, 1.0])
        ylim([0,N])
        ylabel("#", fontsize=10)
        xlabel("Normalized weight", fontsize=10)

        # Weights per activation of pattern's neurons
        subplot(gs[2,1])
        plot(patt_act,weights[patt_range[0]:patt_range[1]],'.')
        ylim([0, 1.0])

        # Show the figure
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
    inputLayer = masquelier()
