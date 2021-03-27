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
    OMEGAFILE = 'power_omega-s-3.txt'
elif data_type == 1:
    name_string = 'cdf'
    OMEGAFILE = 'cdf_omega.txt'
omega_true = np.loadtxt(OMEGAFILE)

# plot properties of data
for (i,N) in enumerate(nsamps):

    # load data
    data = sample_npn(omega_true, data_type, int(N))

    # plot covariance
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(np.log(np.linalg.inv(np.cov(data.T))), cmap = plt.cm.YlGnBu, alpha=.9, aspect='equal', extent=[0.5,dim+.5,dim+.5,0.5])
    ax.grid(None)
    #ax.tick_params(direction='out',bottom=False, left=True)
    ax.set_xticks(np.arange(1,dim+1,1))
    ax.set_yticks(np.arange(1,dim+1,1))
    ax.set_ylim([0.5, dim+0.5])
    im.set_clim(-4,0)
    fig.colorbar(im, ax=ax)
    plt.gca().invert_yaxis()
    plt.savefig('figures/invcov_'+name_string+'_N'+str(int(N))+'_logscale.pdf')
    plt.close(fig)

# plot SING results
for (i,N) in enumerate(nsamps):
    for (l,order) in enumerate(orders):
        for (j,delta) in enumerate(deltas):
            for k in range(MC_runs):

                # load data from file    
                datafile = './outputMC/npn-gp-%s-%d-%d-%s-%d-%d.txt' %(name_string,N,dim,delta_string[j],order,k)
                GP_values = np.loadtxt(datafile, comments='%')

                # plot GP  
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                im = ax.imshow(np.log(GP_values), cmap = plt.cm.YlGnBu, alpha=.9, aspect='equal', extent=[0.5,dim+.5,dim+.5,0.5])
                ax.grid(None)
                ax.set_xticks(np.arange(1,dim+1,1))
                ax.set_yticks(np.arange(1,dim+1,1))
                im.set_clim(-4,0)
                fig.colorbar(im, ax=ax)
                ax.set_ylim([0.5, dim+0.5])
                plt.gca().invert_yaxis()
 
                plt.savefig('figures/gp_'+name_string+'_N'+str(int(N))+'_order'+str(order)+'_delta'+delta_string[j]+'_run'+str(k)+'.pdf')
                plt.close(fig)

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
