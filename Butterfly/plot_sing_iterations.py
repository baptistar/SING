#
# This file runs the algorithm given a set of samples
#
import warnings
import time
import numpy as np
import TransportMaps.Algorithms.SparsityIdentification as SI
from NonGaussianGraphs import Butterfly

import sys
sys.path.append("../SING/")
from SparsityIdentificationNonGaussian import SING

# set parameters
nsamps       = [3000]
deltas	     = [1.]#np.sqrt(2.)]
delta_string = ['1']#sqrt2']
dim          = 10
orders       = [1,2,3]
ordering     = SI.ReverseCholesky()
name_string  = 'butterfly'

# define model
model = Butterfly(dim)

for N in nsamps:

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

    # Apply a random permutation to the data
    rand_perm = np.array([1,2,9,6,3,0,5,4,8,7]) #np.random.permutation(dim)
    processed_data = processed_data[:,rand_perm]
    print("permutation = ", rand_perm)
    np.savetxt('gp_iter/perm.txt',rand_perm,delimiter=',')

    # run SING for each delta
    for order in orders:
        for i, delta in enumerate(deltas):

            recovered_gp = SING(processed_data, order, ordering, delta, plotting=True, delta_str=delta_string[i])
            print("gp = ",recovered_gp)


# -- END OF FILE --
