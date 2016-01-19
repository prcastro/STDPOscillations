from brian import *
def plotpats(times, actv):
    N = len(actv[0])
    discrete_actv=[[] for i in range(N)]
    for n in range(N):
        for ti in range(1,len(times)):
            t = times[ti-1]
            while t < times[ti]:
                discrete_actv[n] += [actv[ti][n]]
                t+=10

    matshow(discrete_actv, cmap=plt.cm.gray)
    show()
