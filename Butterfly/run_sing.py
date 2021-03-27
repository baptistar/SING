#
# This file runs the algorithm given a set of samples
#
import warnings
import time
import numpy as np
import TransportMaps.Algorithms.SparsityIdentification as SI
from NonGaussianGraphs import Butterfly

import sys
sys.path.append("../SING")
from SparsityIdentificationNonGaussian import SING

# set parameters
nsamps       = np.floor(np.logspace(2,4,20)) #np.array([1000,2000])
nsamps       = nsamps[11:]
deltas	     = [1.]#np.sqrt(2.)]
delta_string = ['1']
MC_runs      = 1#25
dim          = 10
orders       = [1,3]
ordering     = SI.ReverseCholesky()
name_string  = 'butterfly'

# define model
model = Butterfly(dim)

for N in nsamps:
    for iter in range(MC_runs):

        # select random subset of data
        data = model.rvs(int(N))

        # Scale the data
        mean_vec = np.mean(data,axis=0)
        print("mean_vec = ",mean_vec)
        data1 = data - mean_vec
        print("shifted data = ",data1)
        inv_var = 1./(np.var(data1,axis=0))
        print("inv var = ",inv_var)
        inv_std = np.diag(np.sqrt(inv_var))
        rescaled_data = np.dot(data1,inv_std)
        print("rescaled data = ", rescaled_data[0:2,:],"\n")
        print("****************************************************\n")
        processed_data = rescaled_data

        # run SING for each delta
        for order in orders:
            for i, delta in enumerate(deltas):

                recovered_gp = SING(processed_data, order, ordering, delta)
                print("gp = ",recovered_gp)
               
                # extract adjacency graph from GP 
                dim = recovered_gp.shape[0]
                adjacency = np.zeros([dim, dim])
                adjacency[recovered_gp != 0] = 1
                
                # save GP and adjacency graph
                np.savetxt('./outputMC/adj-%s-%d-%d-%s-%d-%d.txt' %(name_string,N,dim,delta_string[i],order,iter), adjacency, fmt='%1.3f')
                np.savetxt('./outputMC/gp-%s-%d-%d-%s-%d-%d.txt' %(name_string,N,dim,delta_string[i],order,iter), recovered_gp, fmt='%1.3f')


# -- END OF FILE --
