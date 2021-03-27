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

# plot true graph under permutation
model = Butterfly(dim) 
true_graph = model.edgeSet()
true_graph_perm = true_graph[rand_perm,:][:, rand_perm]
# plot result
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(true_graph_perm, cmap=plt.cm.YlGnBu, alpha=.9, aspect='equal', extent=[0.5,dim+.5,dim+.5,0.5])
ax.grid(None)
ax.tick_params(direction='out',bottom=False, left=True)
ax.set_xticks(np.arange(1,dim+1,1))
ax.set_yticks(np.arange(1,dim+1,1))
ax.set_xticklabels(['$P_1$','$Q_1$','$P_2$','$Q_2$','$P_3$','$Q_3$','$P_4$','$Q_4$','$P_5$','$Q_5$'])
ax.set_yticklabels(['$P_1$','$Q_1$','$P_2$','$Q_2$','$P_3$','$Q_3$','$P_4$','$Q_4$','$P_5$','$Q_5$'])
ax.set_ylim([0.5, dim+0.5])
plt.gca().invert_yaxis()
plt.savefig('gp_iter/true_graph_permuted.pdf')

for counter in range(1,n_iter+1):

    # load GP and apply inverse permutation
    file_name = 'gp_iter/graph_order_' + str(order) + '_N' + str(N) + '_delta' + delta_str + '_iter' + str(counter) + '.txt'
    omegaHat = np.loadtxt(file_name)
    omegaHat = omegaHat[inv_rand_perm,:][:,inv_rand_perm]

    # convert to adjacency
    adjacency_plot = np.zeros([dim, dim])
    adjacency_plot[omegaHat != 0] = 1

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(np.log(omegaHat), vmin=-2.5, vmax=3, cmap = plt.cm.YlGnBu, alpha=.9, aspect='equal', extent=[0.5,dim+.5,dim+.5,0.5])
    ax.grid(None)
    ax.set_xticks(np.arange(1,dim+1,1))
    ax.set_yticks(np.arange(1,dim+1,1))
    ax.set_xticklabels(['$P_1$','$Q_1$','$P_2$','$Q_2$','$P_3$','$Q_3$','$P_4$','$Q_4$','$P_5$','$Q_5$'])
    ax.set_yticklabels(['$P_1$','$Q_1$','$P_2$','$Q_2$','$P_3$','$Q_3$','$P_4$','$Q_4$','$P_5$','$Q_5$'])
    cbar = fig.colorbar(im, ax=ax)
    #cbar.set_clim(-2,3)
    ax.set_ylim([0.5, dim+0.5])
    plt.gca().invert_yaxis()
    plt.savefig('gp_iter/score_order_' + str(order) + '_N' + str(N) + '_delta' + delta_str + '_iter' + str(counter) + '.pdf')
    plt.close(fig)

    # plot result
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(adjacency_plot, cmap=plt.cm.YlGnBu, alpha=.9, aspect='equal', extent=[0.5,dim+.5,dim+.5,0.5])
    ax.grid(None)
    ax.set_xticks(np.arange(1,dim+1,1))
    ax.set_yticks(np.arange(1,dim+1,1))
    ax.set_xticklabels(['$P_1$','$Q_1$','$P_2$','$Q_2$','$P_3$','$Q_3$','$P_4$','$Q_4$','$P_5$','$Q_5$'])
    ax.set_yticklabels(['$P_1$','$Q_1$','$P_2$','$Q_2$','$P_3$','$Q_3$','$P_4$','$Q_4$','$P_5$','$Q_5$'])
    ax.set_ylim([0.5, dim+0.5])
    plt.gca().invert_yaxis()
    plt.savefig('gp_iter/graph_order_' + str(order) + '_N' + str(N) + '_delta' + delta_str + '_iter' + str(counter) + '.pdf')
    plt.close(fig)

# -- END OF FILE --
