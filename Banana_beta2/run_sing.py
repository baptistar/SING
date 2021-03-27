#
# This file runs the algorithm given a set of samples
#
import warnings
import time
import numpy as np
import scipy.io as sio
import TransportMaps.Algorithms.SparsityIdentification as SI
from generate_banana_data import sample_npn_beta2 

import sys
sys.path.append("../SING")
from SparsityIdentificationNonGaussian import SING

# set parameters
#nsamps       = np.floor(np.logspace(2,5,num=20)) 
nsamps       = [10000]
deltas	     = [1.] #np.sqrt(2.)]
delta_string = ['1']
MC_runs      = 1#25
order_vect   = [1,2]
ordering     = SI.ReverseCholesky()
name_string  = 'banana'

for N in nsamps:
    for iter in range(MC_runs):

        # generate data
        data = sample_npn_beta2(int(N))

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

        # run SING for each delta and order
        for order in order_vect:
            for i, delta in enumerate(deltas):

                recovered_gp = SING(processed_data, order, ordering, delta)
                print("gp = ",recovered_gp)
               
                # extract adjacency graph from GP 
                dim = recovered_gp.shape[0]
                adjacency = np.zeros([dim, dim])
                adjacency[recovered_gp != 0] = 1
                
                # save GP and adjacency graph
                np.savetxt('./outputMC/ban_adj-%s-%d-%d-%s-%d-%d.txt' %(name_string,N,dim,delta_string[i],order,iter), adjacency, fmt='%1.3f')
                np.savetxt('./outputMC/ban_gp-%s-%d-%d-%s-%d-%d.txt' %(name_string,N,dim,delta_string[i],order,iter), recovered_gp, fmt='%1.3f')

# -- END OF FILE --
