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
    t=0
    while t<totalTime:
        t += exponential(1250)
        ptimes.append(t) #*ms
    times+=ptimes
    times=sort(times)
    activations = rand(len(times), N)


    for (i,t) in enumerate(times):
        if t in ptimes: #t*second
            activations[i, 0:200] = pattern
    plotpats(times, activations)
    return TimedArray(activations,times)

if __name__ == "__main__":
    patterns(2000, 10000)
