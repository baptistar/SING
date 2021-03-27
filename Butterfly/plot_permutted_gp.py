#
# This file runs the algorithm given a set of samples
#
import warnings
import numpy as np
import matplotlib.pyplot as plt
from NonGaussianGraphs import Butterfly

# set parameters
N            = 3000
delta_str    = '1'
dim          = 10
order        = 3
name_string  = 'butterfly'
n_iter       = 7

# Load random permutation
rand_perm = np.loadtxt('gp_iter/perm.txt',delimiter=',')
rand_perm = rand_perm.astype(int)

# find inverse permutation
inv_rand_perm = np.argsort(rand_perm)

# define axes labels
label = ['$P_1$','$Q_1$','$P_2$','$Q_2$','$P_3$','$Q_3$','$P_4$','$Q_4$','$P_5$','$Q_5$']
label = np.array(label)
label = label[rand_perm]
label = label.tolist()

# plot true graph under permutation
model = Butterfly(dim) 
true_graph = model.edgeSet()
true_graph_perm = true_graph[rand_perm,:][:, rand_perm]
# plot result
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(true_graph_perm, cmap=plt.cm.YlGnBu, alpha=.9, aspect='equal', extent=[0.5,dim+.5,dim+.5,0.5])
ax.grid(None)
ax.tick_params(direction='out',bottom=True, left=True)
ax.set_xticks(np.arange(1,dim+1,1))
ax.set_yticks(np.arange(1,dim+1,1))
ax.set_xticklabels(label) #['$P_1$','$Q_1$','$P_2$','$Q_2$','$P_3$','$Q_3$','$P_4$','$Q_4$','$P_5$','$Q_5$'])
ax.set_yticklabels(label) #['$P_1$','$Q_1$','$P_2$','$Q_2$','$P_3$','$Q_3$','$P_4$','$Q_4$','$P_5$','$Q_5$'])
ax.set_ylim([0.5, dim+0.5])
plt.gca().invert_yaxis()
plt.savefig('gp_iter/true_graph_permuted_relabel.pdf')

# -- END OF FILE --
