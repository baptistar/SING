#
# This file runs the algorithm given a set of samples
#
import warnings
import time
import numpy as np
import scipy.io as sio
import TransportMaps.Algorithms.SparsityIdentification as SI

import sys
sys.path.append("../SING")
from SparsityIdentificationNonGaussian import SING

# define parameters
nsamps       = [3000]
name_string  = 'L96'
order_vect   = [1,2]
d            = 15
dt_iter      = 40
MC_runs      = 5
ordering     = SI.ReverseCholesky()
deltas       = [1.]#np.sqrt(2.)]
delta_string = ['1']#sqrt2']
offset       = 0.1
offset_str   = '0.1'

# load data from file
DATAFILE = 'data/L96_d'+str(d)+'_dt'+str(dt_iter)+'.mat'
data_all = sio.loadmat(DATAFILE)
data_all = data_all['data'].T

for N in nsamps:
    for iter in range(MC_runs):

        # select random subset of data
        rand_samples = np.random.randint(0,data_all.shape[0],int(N))
        data = data_all[rand_samples,:]
        print('data.shape = ',data.shape)

        # scale the data
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
        for order in order_vect:
            for i, delta in enumerate(deltas):
                recovered_gp = SING(processed_data, order, ordering, delta, offset=offset)
                print("gp = ",recovered_gp)
                
                # extra adjacency matrix from GP
                dim = recovered_gp.shape[0]
                adjacency = np.zeros([dim, dim])
                adjacency[recovered_gp != 0] = 1
                
                # save GP and adjacency graph
                np.savetxt('./outputMC/adj-%s-%d-%d-%s-%s-%d-%d.txt' %(name_string,N,dim,delta_string[i],offset_str,order,iter), adjacency, fmt='%1.3f')
                np.savetxt('./outputMC/gp-%s-%d-%d-%s-%s-%d-%d.txt' %(name_string,N,dim,delta_string[i],offset_str,order,iter), recovered_gp, fmt='%1.3f')
            
# -- END OF FILE --
