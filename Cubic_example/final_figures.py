# post process the mc study data of the sparse maps from samples

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import matplotlib
from matplotlib import rc
from math import *

rc('text',usetex=True)
font={'family' : 'normal',
    'weight' : 'normal',
    'size' :16}
matplotlib.rc('font',**font)

# define parameters
nsamps       = np.floor(np.logspace(2,6,num=25))
deltas	     = [1.0]
delta_string = ['1'] #sqrt2']
name_string  = 'cubic'
dim          = 3
order_vect   = [1,2]
MC_runs      = 25

# extract true precision
omega_true = np.array([[1, 0.2, 0],[0.2, 1, 0.2],[0, 0.2, 1]])
graph_true = np.zeros([dim, dim])
graph_true[omega_true != 0] = 1

# define arrays to store errors and success probability
type1_err = np.zeros((len(nsamps),MC_runs))
type2_err = np.zeros((len(nsamps),MC_runs))
succ_prob = np.zeros((len(nsamps),MC_runs))

# load data from files
for order in order_vect:
    for (j,delta) in enumerate(deltas):
        for (i,N) in enumerate(nsamps):
            for k in range(MC_runs):
        
                # load data from file    
                datafile = './outputMC/npn-adj-%s-%d-%d-%s-%d-%d.txt' %(name_string,N,dim,delta_string[j],order,k)
                graph_approx = np.loadtxt(datafile, comments='%')
        
                # compute type 1 (false positive) and type 2 (false negative) errors
                # Type 1 - asserting extra edge that is absent; extra edges in graph_approx
                # Type 2 - fail to assert edge is present; missed in graph_approx
                type1_err[i,k] = np.sum(np.tril(graph_approx - graph_true > 0))
                type2_err[i,k] = np.sum(np.tril(graph_approx - graph_true < 0))
        
                # compute success probability
                if (type1_err[i,k] == 0 and type2_err[i,k] == 0):
                    succ_prob[i,k] = 1
        
        # plot results
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.errorbar(nsamps, np.mean(type1_err,axis=1), yerr=1.96*np.std(type1_err,axis=1)/np.sqrt(MC_runs), fmt='-o', label='Type 1 errors')
        ax.errorbar(nsamps, np.mean(type2_err,axis=1), yerr=1.96*np.std(type2_err,axis=1)/np.sqrt(MC_runs), fmt='-o', label='Type 2 errors')
        ax.set_xlabel('Sample size, n')
        ax.set_ylabel('Number of edges')
        ax.set_xlim((100,1e6))
        ax.set_ylim((-0.2,2.5))
        ax.set_xscale('log')
        ax.legend(loc='upper right')
        plt.savefig('figures/'+name_string+'_errors_ord'+str(order)+'_delta'+delta_string[j]+'.pdf')
        
