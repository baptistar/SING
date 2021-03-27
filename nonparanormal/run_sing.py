#
# This file runs the algorithm given a set of samples
#
import warnings
import time
import numpy as np
import TransportMaps.Algorithms.SparsityIdentification as SI
import os

import sys
sys.path.append("../SING")
from SparsityIdentificationNonGaussian import SING
from generate_npn_data import sample_npn 

# set parameters
#nsamps       = np.floor(np.logspace(2,5,num=20)) #3000
nsamps       = [3000]
deltas	     = [1.] #[np.sqrt(2.)]
delta_string = ['1'] #['sqrt2']
MC_runs      = 1#25
order_vect   = [1,2]
ordering     = SI.ReverseCholesky()
data_type    = 1

# set data parameters and load omega
if data_type == 0:
    name_string = 'power'
    OMEGAFILE = 'power_omega-s-3.txt'
elif data_type == 1:
    name_string = 'cdf'
    OMEGAFILE = 'cdf_omega.txt'
omega_true = np.loadtxt(OMEGAFILE)

for N in nsamps:
    for iter in range(MC_runs):

        # check if file exists
        file_name = './outputMC/npn-adj-'+name_string+'-'+str(int(N))+'-'+str(10)+'-'+delta_string[-1]+'-'+str(order_vect[-1])+'-'+str(iter)+'.txt'
        if os.path.isfile(file_name):
            continue

        # sample data
        data = sample_npn(omega_true, data_type, int(N))

        # Scale the data
        mean_vec = np.mean(data,axis=0)
        print("mean_vec = ",mean_vec)
        data1 = data - mean_vec
        #print("shifted data = ",data1)
        inv_var = 1./(np.var(data1,axis=0))
        print("inv var = ",inv_var)
        inv_std = np.diag(np.sqrt(inv_var))
        rescaled_data = np.dot(data1,inv_std)
        print("rescaled data = ", rescaled_data[0:2,:],"\n")
        print("****************************************************\n")
        processed_data = rescaled_data

        # run SING for each delta
        for order in order_vect:
            for i, delta in enumerate(deltas):

                recovered_gp = SING(processed_data, order, ordering, delta)
                print("gp = ",recovered_gp)
               
                # extract adjacency graph from GP 
                dim = recovered_gp.shape[0]
                adjacency = np.zeros([dim, dim])
                adjacency[recovered_gp != 0] = 1
                
                # save GP and adjacency graph
                np.savetxt('./outputMC/npn-adj-%s-%d-%d-%s-%d-%d.txt' %(name_string,N,dim,delta_string[i],order,iter), adjacency, fmt='%1.3f')
                np.savetxt('./outputMC/npn-gp-%s-%d-%d-%s-%d-%d.txt' %(name_string,N,dim,delta_string[i],order,iter), recovered_gp, fmt='%1.3f')


# -- END OF FILE --
