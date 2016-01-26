from brian import *
import random as rd
import matplotlib.gridspec as gridspec

# TODO
# [*] record 2000 neurons spiking with constant drive
# [*] sinusoidal drive
# [*] stimulus
# [*] synapse input layer -> output layer
# [*] Record voltage of output layer
# [*] STDP
# [*] Measure learning
# [*] Learn
# [ ] More realistic model of CA1
# [ ] Parameter search
# [ ] Introduce more patterns

def plotActivations(values, times, pattern_presence):
    N = len(values[0])
    numPatterns = len(pattern_presence)
    discrete_actv=[[] for i in range(N)]
    presence = [[] for i in range(20)]
    for n in range(N):
        print(n)
        for ti in range(1,len(times)):
            t = times[ti-1]
            while t < times[ti]:
                discrete_actv[n] += [values[ti-1][n]]
                if n in range(20):
                    signal = sum(pattern_presence[p][ti-1] for p in range(numPatterns))
                    presence[n]  += [1 - (1.0/(numPatterns)) * signal]
                t+=0.01 # In seconds

    matshow(discrete_actv + presence, cmap=plt.cm.gray)
    show()

def NEWactivationLevels(N, totalTime, Npatt, patt_range, toPlot=True):
    '''This function return the activation levels matrix with
    the level of activation of each neuron over time. This is returned as
    a TimedArray. The list of the times in each the pattern in present is
    returned as well'''
    mean_dt   = 250e-3
    mean_dpat = 1250e-3

    patterns = rand(200,Npatt)
    patt_ranges = [(1500,1700),(1700,1900),(1800,2000)]


    # Time intervals
    times = []
    t = 0
    while t < totalTime:
        times.append(t)
        t+= np.random.exponential(mean_dt)
    times.append(totalTime)
    len_t = len(times)

    #creating patterns
    pattern_times = [rd.sample(range(len_t), len_t//5) for i in range(Npatt)]
    pattern_presence = array([ [ 1 if j in pattern_times[i][:] else 0  for j in range(len_t)] for i in range(Npatt)])


    # Random activation matrix - the last 21 rows are for pattern
    #  identification (grey when present and white when not)
    activations = rand(len_t, N)
    times=array(times)

    # Add patterns to activation matrix, and also the identification at
    #  the bottom
    for patti, ptimeis in enumerate(pattern_presence):
        for i in range(len_t):
            if ptimeis[i]:
                activations[i, patt_ranges[patti][0]:patt_ranges[patti][1]] = patterns[:,patti]

    # Make the activations' TimedArray
    activations = TimedArray(activations,times)

    # Transform pattern presence array into pattern intervals
    pattern_intervals=[]
    for i in range(Npatt):
        starts  = activations.times[pattern_presence[i] == 1]
        shifted = array([0] + list(pattern_presence[i])[:-1])
        ends    = activations.times[shifted == 1]
        pattern_intervals += [zip(starts, ends)]

    # Plot activation matrix
    if toPlot:
        plotActivations(activations, times, pattern_presence)

    return activations, pattern_intervals

def activationLevels(N, totalTime, pattern, patt_range, toPlot=True):
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
    pattern_presence = zeros(len(times))

    # Add patterns to activation matrix, and also the identification at
    #  the bottom
    for (i,t) in enumerate(times):
        if t in ptimes:
            activations[i, patt_range[0]:patt_range[1]] = pattern
            pattern_presence[i] = 1

    # Make the activations' TimedArray
    activations = TimedArray(activations,times)

    # Transform pattern presence array into pattern intervals
    starts  = activations.times[pattern_presence == 1]
    shifted = array([0] + list(pattern_presence)[:-1])
    ends    = activations.times[shifted == 1]
    pattern_intervals = zip(starts, ends)

    if toPlot:
        plotActivations(activations, times, pattern_presence)

    return activations, pattern_intervals

def intersection(a,b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def insideInterval(x, I):
    return x > I[0] and x < I[1]

def mutualInformation(init,end, pattern_intervals, spiketimes):
    ts = 0.125 # Timestep 125 milisseconds
    spiketimes /= second
    bins = zip(arange(init, end, ts),arange(init+ts, end+ts, ts))
    patt  = []
    spike = []
    for b in bins:
        patt  += [sum( intersection(b, pati) for pati in pattern_intervals ) >= ts/2.]
        spike += [sum(insideInterval(spk, b) for spk in spiketimes) >= 1]
    patt  = np.array(patt).astype(float) ; npatt  = abs(patt -1)  # s, ns
    spike = np.array(spike).astype(float); nspike = abs(spike -1) # r, nr

    # Probabilities
    pr    = mean(spike)                   ; pnr   = 1 - pr
    ps    = mean(patt)                    ; pns   = 1 - ps
    prs   = dot(spike , patt) / len(patt) ; pnrs  = dot(nspike, patt) / len(patt);
    prns  = dot(spike , npatt)/ len(patt) ; pnrns = dot(nspike, npatt)/ len(patt)

    MI = 0
    if prs   != 0: MI += prs   * log2(prs   / (pr*ps)  )
    if pnrs  != 0: MI += pnrs  * log2(pnrs  / (pnr*ps) )
    if prns  != 0: MI += prns  * log2(prns  / (pr*pns) )
    if pnrns != 0: MI += pnrns * log2(pnrns / (pnr*pns))

    return MI

def masquelier(simTime=1000*second, N=2000, Vt=-54*mV, Vr=-60*mV, El=-70*mV, tau=20*ms, taus=5*ms, taup=16.8*ms, taum=33.7*ms, aplus=0.005, aratio=1.48, R=(10**6)*ohm, oscilFreq=8, patt_act=rand(200), patt_range =(1800,2000), toPlot=True, MIstep=30*second):
    '''This file executes the simulations (given the parameters),
    of Masquelier's model for learning and saves the results in
    appropriate files.

    Note: R is nine times larger than in the paper'''

    if len(patt_act) != (patt_range[1] - patt_range[0]):
        raise ValueError("Pattern activation must be consistent with pattern range")

    # Default timestep and number of steps
    dt = defaultclock.dt

    # Size of noise, drives and weights
    sigma = 0.015*(Vt - Vr)
    Ithr  = (Vt - El)/R
    Imax  = 0.05*namp
    wmax  = 2*(8.6 * pamp/Imax)

    # Model of input neuron. Current comes from oscillatory input (I) and from
    #  the activation levels (actValue, like Masquelier et al 2009 Figure 1). This
    #  is based on Eq 1 of Masquelier et al 2009.
    neuronModel =  Equations('''
    dV/dt = -(V-El - R*(I+actValue+(s*Imax)))/tau + sigma*xi/(tau**0.5): volt
    I: amp
    actValue: amp
    ds/dt = -s/taus: 1
    ''')

    # Create neuron groups and set initial conditions
    inputLayer    = NeuronGroup(N=N, model=neuronModel, threshold=Vt, reset=Vr)
    inputLayer.V  = Vr + rand(N)*(Vt - Vr) # Initial voltage
    outputLayer   = NeuronGroup(N=1, model=neuronModel, threshold=Vt, reset=Vr)
    outputLayer.V = Vr + rand()*(Vt - Vr) # Initial voltage

    # Oscillatory drive on the input layer
    oscilAmp     = 0.15*Ithr
    inputLayer.I = TimedArray((oscilAmp/2)*sin(2*pi*oscilFreq*dt*arange(float(simTime/dt)) - pi))

    # Get the activation levels' matrix and use as input current
    acts, pattern_intervals = NEWactivationLevels(N, simTime/second, 3, patt_range, toPlot=True)
    inputLayer.actValue = (acts*0.12 + 0.95)*Ithr # Affine mapping between activation and input

    # Connect the layers
    weights = rand(N, 1) * wmax
    con     = Connection(inputLayer, outputLayer, 's', weight=weights)

    # STDP synapse
    aminus = -(aplus * aratio)
    stdp   = ExponentialSTDP(con, taup, taum, aplus, aminus, wmax=1, interactions='all', update='additive')

    # Mesurement devices
    spikes_input  = SpikeMonitor(inputLayer[patt_range[0]-50:patt_range[1]-50])
    spikes_output = SpikeMonitor(outputLayer)
    voltimeter    = StateMonitor(outputLayer, 'V', record=0)
    amperimeter   = StateMonitor(inputLayer, 'I', record=0)

    # Run the simulation
    # Runs for initialStep time
    initialStep=100*second
    run(initialStep, report='text')

    # Will run and calculate MI for each MIstep
    MIs     = [0]
    nowTime = int(initialStep)
    while nowTime*second < simTime:
        init = int(nowTime)
        run(MIstep, report='text')
        nowTime += int(MIstep)
        MIs += [mutualInformation(init, nowTime, pattern_intervals, spikes_output[0])]
        print('from '+str(init)+' to ' +str(nowTime))
        print(str(MIs[-1])+" bits")
        # If the change in the last two iterations was < 1e-3, stop simulating
        if len(MIs) > 3 and MIs[-1]-MIs[-2]<1e-3 and MIs[-2]-MIs[-3]<1e-3:
            print("Changes in MI too small, stopping simulation")
            break

    if toPlot:
        st    = simTime/second
        xlims = array([ (st>5)*(st-5), st])
        # Update weights to the values found on the Connection object
        weights = con.W.todense()

        # Plot raster + voltage of neuron 0
        raster_voltage = figure(1)

        # Set grid
        gs = gridspec.GridSpec(3, 2)
        gs.update(hspace=0.9)

        # Raster plot
        subplot(gs[0,:])
        raster_plot(spikes_input, markersize=4, color='k')
        # Plot grey stripe on spike intervals
        for start, end in pattern_intervals:
            axvspan(start*1000, end*1000, color='grey', alpha=0.5, lw=0)
        xlim(xlims*1e3)
        ylabel('Afferent #')
        xlabel('Time (in ms)', fontsize=10)

        # Plot membrane potential of the output with patterns
        subplot(gs[1,:])
        plot(voltimeter.times/second, voltimeter[0]/mV, color='k')
        axhline(-54, linestyle=':', color='k')
        # Plot action potentials
        for t in spikes_output[0]:
            axvline(t, linestyle=':', color='k')
        # Plot grey stripe on spike intervals
        for start, end in pattern_intervals:
            axvspan(start, end, color='grey', alpha=0.5, lw=0)
        xlim(xlims)
        ylim(-70, -53)

        xlabel('Time (in s)', fontsize=10)
        ylabel('Membrane potential (in mV)', fontsize=10)

        # Weights' histogram
        subplot(gs[2,0])
        hist(weights, 9, color='k')
        xlim(0.0, 1.0)
        ylim(0,N)
        ylabel('#', fontsize=10)
        xlabel('Normalized weight', fontsize=10)

        # Weights per activation of pattern's neurons
        subplot(gs[2,1])
        plot(patt_act, weights[patt_range[0]:patt_range[1]], '.', color='k')
        ylim(0, 1.0)
        ylabel('Weight')
        xlabel('Pattern activation level')

        # Show the figure
        raster_voltage.show()

        # Save figure
        raster_voltage.savefig('summary' + '_f' + str(oscilFreq) + '_aratio' + str(aratio) + '_t' + str(simTime/second) +'_N' +str(N) +'.png')

    return inputLayer, MIs

if __name__ == "__main__":
    inputLayer, MI = masquelier(simTime = 3*second, MIstep=30*second, R = 7.9e6*ohm)
