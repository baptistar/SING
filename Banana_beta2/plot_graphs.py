# post process the mc study data of the sparse maps from samples

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import matplotlib
from matplotlib import rc
from math import *

rc('text',usetex=True)
font={'family' : 'normal',
    'weight' : 'bold',
    'size' :20}
matplotlib.rc('font',**font)

# define parameters
nsamps       = np.array([1e4])
deltas	     = [1]
delta_string = ['1']
name_string  = 'banana'
dim          = 5
orders       = [1,2]
MC_runs      = 1

# extract true precision
graph_true = np.eye(dim)
graph_true[0:dim,0] = np.ones((1,dim))
graph_true[0,1:dim] = np.ones((1,dim-1))

# plot true graph
max_graph = np.max(graph_true)
graph_plot = (1/max_graph)*graph_true

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(graph_plot, cmap=plt.cm.YlGnBu, alpha=.9, aspect='equal', extent=[0.5,dim+.5,dim+.5,0.5])
ax.grid(None)
ax.tick_params(direction='out',bottom=True, left=True)
ax.set_xticks(np.arange(1,dim+1,1))
ax.set_yticks(np.arange(1,dim+1,1))
ax.set_ylim([0.5, dim+0.5])
plt.gca().invert_yaxis()
plt.savefig('figures/true_graph.pdf')

# load graph data from files
for (i,N) in enumerate(nsamps):
    for (l,order) in enumerate(orders):
        for (j,delta) in enumerate(deltas):
            for k in range(MC_runs):

                # load data from file    
                datafile = './outputMC/ban_adj-%s-%d-%d-%s-%d-%d.txt' %(name_string,N,dim,delta_string[j],order,k)
                graph_approx = np.loadtxt(datafile, comments='%')
                max_graph = np.max(graph_approx)
                graph_plot = (1/max_graph)*graph_approx

                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                im = ax.imshow(graph_plot, cmap = plt.cm.YlGnBu, alpha=.9, aspect='equal', extent=[0.5,dim+.5,dim+.5,0.5])
                ax.grid(None)
                ax.tick_params(direction='out',bottom=True, left=True)
                ax.set_xticks(np.arange(1,dim+1,1))
                ax.set_yticks(np.arange(1,dim+1,1))
                ax.set_ylim([0.5, dim+0.5])
                plt.gca().invert_yaxis()

                plt.savefig('figures/graph_N'+str(int(N))+'_order'+str(order)+'_delta'+delta_string[j]+'_run'+str(k)+'.pdf')
                plt.close(fig)
