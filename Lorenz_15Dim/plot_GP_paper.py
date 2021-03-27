#
# This file runs the algorithm given a set of samples
#
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
from matplotlib import rc
from os import path

rc('text',usetex=True)
font={'family' : 'normal',
    'weight' : 'normal',
    'size' :12}
matplotlib.rc('font',**font)

# define parameters
nsamps       = [3000]
name_string  = 'L96'
orders       = [1,2]
dim          = 15
MC_runs      = 5
delta        = [1.0] #np.sqrt(2.)]
delta_string = ['1'] #'sqrt2']
offset_str   = '0.1'
offset_str_plot = '0p1'

# extract GP results for each order, N, and MC iteration
for (i,order) in enumerate(orders):
    for (j,N) in enumerate(nsamps):
        for iter in range(MC_runs):
            for (k,delt) in enumerate(delta):

                # check if file exists
                file_name = name_string+'-'+str(int(N))+'-'+str(dim)+'-'+delta_string[k]+'-'+offset_str+'-'+str(order)+'-'+str(iter)+'.txt'
                if (path.exists('./outputMC/gp-'+file_name) == False):
                    continue

                GP_values  = np.loadtxt('./outputMC/gp-'+file_name, comments='%')
                Adj_values = np.loadtxt('./outputMC/adj-'+file_name, comments='%')

                # plot GP  
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                im = ax.imshow(np.log(GP_values), cmap = plt.cm.YlGnBu, alpha=.9, aspect='equal', extent=[0.5,dim+.5,dim+.5,0.5])
                ax.grid(None)
                ax.set_xticks(np.arange(1,dim+1,1))
                ax.set_yticks(np.arange(1,dim+1,1))
                fig.colorbar(im, ax=ax)
                ax.set_ylim([0.5, dim+0.5])
                plt.gca().invert_yaxis()
 
                plt.savefig("./figures/gp-%s-%d-%d-%s-%s-%d-%d.pdf" %(name_string,N,dim,delta_string[k],offset_str_plot,order,iter))
                plt.close(fig)
              
                # plot adjacency matrix 
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                im = ax.imshow(Adj_values, cmap = plt.cm.YlGnBu, alpha=.9, aspect='equal', extent=[0.5,dim+.5,dim+.5,0.5])
                ax.grid(None)
                ax.set_xticks(np.arange(1,dim+1,1))
                ax.set_yticks(np.arange(1,dim+1,1))
                #fig.colorbar(im, ax=ax)
                ax.set_ylim([0.5, dim+0.5])
                plt.gca().invert_yaxis()
                
                plt.savefig("./figures/adj-%s-%d-%d-%s-%s-%d-%d.pdf" %(name_string,N,dim,delta_string[k],offset_str_plot,order,iter))
                plt.close(fig)


# -- END OF FILE --
