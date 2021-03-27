#
# This file runs the algorithm given a set of samples
#
import warnings
import time
import numpy as np
import TransportMaps.Algorithms.SparsityIdentification as SI

import sys
sys.path.append("../SING")
from SparsityIdentificationNonGaussian import SING
import os

# set parameters
nsamps       = np.floor(np.logspace(2,6,num=25))
deltas	     = [1.0]#np.sqrt(2.0)]
delta_string = ['1']##sqrt2']
MC_runs      = 25
order_vect   = [1,2]
ordering     = SI.ReverseCholesky()
dim          = 3
name_string = 'cubic'

# define inverse covariance and covariance
Sigma_inv = [[1, 0.2, 0],[0.2, 1, 0.2],[0, 0.2, 1]]
Sigma = np.linalg.inv(Sigma_inv)

for N in nsamps:
    for iter in range(MC_runs):

        # check if file exists
        file_name = './outputMC/npn-adj-'+name_string+'-'+str(int(N))+'-'+str(3)+'-'+delta_string[-1]+'-'+str(order_vect[-1])+'-'+str(iter)+'.txt'
        if os.path.isfile(file_name):
            continue

        # generate random samples
        normal_data = np.random.multivariate_normal(np.zeros(dim), Sigma, size=(int(N)))
        data = normal_data**3

        # Scale the data
        mean_vec = np.mean(data,axis=0)
        #print("mean_vec = ",mean_vec)
        data1 = data - mean_vec
        #print("shifted data = ",data1)
        inv_var = 1./(np.var(data1,axis=0))
        #print("inv var = ",inv_var)
        inv_std = np.diag(np.sqrt(inv_var))
        rescaled_data = np.dot(data1,inv_std)
        #print("rescaled data = ", rescaled_data[0:2,:],"\n")
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
