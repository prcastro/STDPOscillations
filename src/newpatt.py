from brian import *
from plotpats import *
def patterns(N, totalTime):
    pattern= rand(200)

    times = []
    t = 0
    while t < totalTime:
        times.append(t) #*ms
        t+= exponential(250)

    times.append(totalTime) #*ms
    ptimes = []
    t=exponential(1250)
    while t<totalTime:
        ptimes.append(t) #*ms
        t += exponential(1250)
    times+=ptimes
    times=sort(times)
    activations = rand(len(times), N+21)


    for (i,t) in enumerate(times):
        if t in ptimes: #t*second
            activations[i, 1800:2000] = pattern
            activations[i,N:N+20] = .8
        else:
            activations[i,N:N+20] = 1
    plotpats(times, activations)
    return TimedArray(activations,times)

if __name__ == "__main__":
    patterns(2000, 10000)
