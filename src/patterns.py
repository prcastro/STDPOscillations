from brian import *
def patterns(ngroup, simTime, Npatt_neurons, act_lims, Npats=1):
    '''This function changes cI values of a given group of neurons,

    '''
    N = len(ngroup)
    #pat_neurons=range(Npatt_neurons) #the first Npatt_neurons are the ones related to the pattern
    pat_act = [ [(2*i)*(act_lims[1]-act_lims[0])+act_lims[0] for j in range(Npatt_neurons)] for i in range(Npats) ]
    # After getting results = masquelier, use this:
    ###############
    pat_neurons = [ [ [] for j in range(Npatt_neurons) ] for i in range(Npats)]
    for i in range(Npats):
        pat_neurons[i][:] = range(i*Npatt_neurons,(i+1)*Npatt_neurons)#randint(0,N,Npatt_neurons)
    pat_neurons=array(pat_neurons)
    print(pat_neurons)
    ###############

    # 250 is the mean of the exp. distribution, and //100 is the number of 100ms steps needed to reach simTim
    Nsteps= 4# int(simTime/ms)//100 #number of change steps calculated
    Npatstep = Nsteps#//5 #number of pattern steps calculated

    #timesteps = exponential(100, Nsteps)
    #patsteps = exponential(50, Npatstep)
    timesteps = array([10, 240, 100,400])
    patsteps = array([50,250,190,400])

    for i in range(1,Nsteps):
        timesteps[i] = timesteps[i] + timesteps[i-1]
    print(timesteps)
    for i in range(1,Npatstep):
        patsteps[i]  = patsteps[i]  + patsteps[i-1]
    print(patsteps)
    #putting all changes in same list alltimes
    alltimes = sort( list(timesteps)+list(patsteps) )
    print(alltimes)
    # array of ones and zeros, being one when there is a pattern
    pat_occurs =sum([array(alltimes) == patsteps[i] for i in range(len(patsteps))], 0)
    #dumping everything bigger than simTime
    last_index = argmax(alltimes > int(simTime/ms)) #gets the first >simtime
    alltimes = alltimes[:last_index]
    print(last_index)
    print(alltimes)
    print(pat_occurs)
    for neuroni in range(N):
        act_values = [ rand()*(act_lims[1] - act_lims[0]) + act_lims[0] for t in alltimes]
        for p in range(Npats):
            if neuroni in pat_neurons[p]:
                for t in range(len(act_values)):
                    if pat_occurs[t]: #if there is a pattern at time t
                        specific_act = pat_act[p][argmax(pat_neurons[p]==neuroni)]
                        act_values[t] = specific_act
        ngroup[neuroni].cI = TimedArray(act_values, times= alltimes)

    return ngroup
