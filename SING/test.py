import numpy as np
np.random.seed(1)

from SparsityIdentificationNonGaussian import SING
from NodeOrdering import ReverseCholesky

# set parameters
N            = 1e4
delta 	     = 3.
order        = 2
ordering     = ReverseCholesky()
dim          = 3
name_string = 'cubic'

# define inverse covariance and covariance
Sigma_inv = [[1, 0.2, 0],[0.2, 1, 0.2],[0, 0.2, 1]]
Sigma = np.linalg.inv(Sigma_inv)

# generate random samples
normal_data = np.random.multivariate_normal(np.zeros(dim), Sigma, size=(int(N)))
data = normal_data**3

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
recovered_gp = SING(processed_data, order, ordering, delta)
print("gp = ",recovered_gp)

