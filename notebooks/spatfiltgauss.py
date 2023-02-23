
import numpy as np

def spatfiltergauss(vin, grid, d, grid2):


    ng, ns = np.shape(vin)
    alpha = d/np.sqrt(-np.log(.5))

    ng2, ndum = np.shape(grid2)
    vout = np.zeros([ng2, ns])

    for i in range(0, ng2):

        r0 = grid2[i, :]
        rd = grid - np.tile(r0, (ng, 1))
        dis = np.sqrt(np.sum(rd**2, axis=1))
        w = exp(-dis**2/alpha**2)

        for j in range(0, ns):

            vout(i,j)=np.sum(w*vin[:, j]), axis=0)/sum(w, axis=0);

    return vout
