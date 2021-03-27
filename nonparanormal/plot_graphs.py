# post process the mc study data of the sparse maps from samples

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import matplotlib
from matplotlib import rc
from math import *
from generate_npn_data import sample_npn 

rc('text',usetex=True)
font={'family' : 'normal',
    'weight' : 'bold',
    'size' :20}
matplotlib.rc('font',**font)

# define parameters
nsamps       = np.array([3000])
deltas	     = [1]
delta_string = ['1']
dim          = 10
orders       = [1,2]
MC_runs      = 1
data_type    = 1

# extract true graph
if data_type == 0:
    name_string = 'power'
    OMEGAFILE = '../../data_samples/copula_data/power_omega-s-3.txt'
elif data_type == 1:
    name_string = 'cdf'
    OMEGAFILE = '../../data_samples/copula_data/cdf_omega.txt'
omega_true = np.loadtxt(OMEGAFILE)
graph_true = np.zeros([dim, dim])
graph_true[omega_true != 0] = 1

# plot true graph
max_graph = np.max(graph_true)
graph_plot = (1/max_graph)*graph_true

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(graph_plot, cmap=plt.cm.YlGnBu, alpha=.9, aspect='equal', extent=[0.5,dim+.5,dim+.5,0.5])
ax.grid(None)
#ax.tick_params(direction='out',bottom=False, left=True)
ax.set_xticks(np.arange(1,dim+1,1))
ax.set_yticks(np.arange(1,dim+1,1))
ax.set_ylim([0.5, dim+0.5])
plt.gca().invert_yaxis()
plt.savefig('figures/true_graph_'+name_string+'.pdf')

# plot properties of data
hist_idx = 0
for (i,N) in enumerate(nsamps):

    # load data
    data = sample_npn(omega_true, data_type, int(N))

    # plot covariance
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(np.linalg.inv(np.cov(data.T)), cmap = plt.cm.YlGnBu, alpha=.9, aspect='equal', extent=[0.5,dim+.5,dim+.5,0.5])
    ax.grid(None)
    #ax.tick_params(direction='out',bottom=False, left=True)
    ax.set_xticks(np.arange(1,dim+1,1))
    ax.set_yticks(np.arange(1,dim+1,1))
    ax.set_ylim([0.5, dim+0.5])
    plt.gca().invert_yaxis()
    plt.savefig('figures/cov_'+name_string+'_N'+str(int(N))+'.pdf')
    plt.close(fig)

    # plot histogram
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.grid(None)
    plt.hist(data[:,hist_idx], density=True, bins=100)
    plt.xlabel('$Z_'+str(hist_idx+1)+'$')
    ax.set_xticks(np.array([-1,0,1]))
    ax.set_xticklabels(['$-1$','$0$','$1$'])
    plt.tight_layout()
    plt.savefig('figures/histogram_'+name_string+'_N'+str(int(N))+'.pdf')
    plt.close(fig)


# plot SING results
for (i,N) in enumerate(nsamps):
    for (l,order) in enumerate(orders):
        for (j,delta) in enumerate(deltas):
            for k in range(MC_runs):

                # load data from file    
                datafile = './outputMC/npn-adj-%s-%d-%d-%s-%d-%d.txt' %(name_string,N,dim,delta_string[j],order,k)
                graph_approx = np.loadtxt(datafile, comments='%')
                max_graph = np.max(graph_approx)
                graph_plot = (1/max_graph)*graph_approx

                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                im = ax.imshow(graph_plot, cmap = plt.cm.YlGnBu, alpha=.9, aspect='equal', extent=[0.5,dim+.5,dim+.5,0.5])
                ax.grid(None)
                #ax.tick_params(direction='out',bottom=False, left=True)
                ax.set_xticks(np.arange(1,dim+1,1))
                ax.set_yticks(np.arange(1,dim+1,1))
                ax.set_ylim([0.5, dim+0.5])
                plt.gca().invert_yaxis()

                plt.savefig('figures/graph_'+name_string+'_N'+str(int(N))+'_order'+str(order)+'_delta'+delta_string[j]+'_run'+str(k)+'.pdf')
                plt.close(fig)
