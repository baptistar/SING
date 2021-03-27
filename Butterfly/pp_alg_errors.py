# post process the mc study data of the sparse maps from samples

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import matplotlib
from matplotlib import rc
from math import *
import sys
sys.path.append("../../graphs/")
from NonGaussianGraphs import Butterfly
from os import path

rc('text',usetex=True)
font={'family' : 'normal',
    'weight' : 'normal',
    'size' :12}
matplotlib.rc('font',**font)

# define parameters
nsamps       = np.floor(np.logspace(2,4,10)) #np.floor(np.logspace(2,4,num=5))
deltas	     = [2,3]
delta_string = ['2','3']
name_string  = 'butterfly'
dim          = 10
order_vect   = [1,2,3]
MC_runs      = 1

# extract true precision
model = Butterfly(dim)
graph_true = model.edgeSet()

# load data from files
for order in order_vect:

    # define arrays to store errors and success probability
    type1_err = np.zeros((len(nsamps),len(deltas),MC_runs))
    type2_err = np.zeros((len(nsamps),len(deltas),MC_runs))
    succ_prob = np.zeros((len(nsamps),len(deltas),MC_runs))
    type1_err.fill(np.nan)
    type2_err.fill(np.nan)
    succ_prob.fill(np.nan)

    for (i,N) in enumerate(nsamps):
        for (j,delta) in enumerate(deltas):
            for k in range(MC_runs):
    
                # load data from file    
                datafile = './outputMC/adj-%s-%d-%d-%s-%d-%d.txt' %(name_string,N,dim,delta_string[j],order,k)
                if (path.exists(datafile) == False):
                    continue
                graph_approx = np.loadtxt(datafile, comments='%')
    
                # compute type 1 (false positive) and type 2 (false negative) errors
                # Type 1 - asserting extra edge that is absent; extra edges in graph_approx
                # Type 2 - fail to assert edge is present; missed in graph_approx
                type1_err[i,j,k] = np.sum(np.tril(graph_approx - graph_true > 0))
                type2_err[i,j,k] = np.sum(np.tril(graph_approx - graph_true < 0))
    
                # compute success probability
                if (type1_err[i,j,k] == 0 and type2_err[i,j,k] == 0):
                    succ_prob[i,j,k] = 1 
                else:
                    succ_prob[i,j,k] = 0
    
    # plot results
    fig, axs = plt.subplots(1,3, figsize=(14,5))
    
    for (j,delta) in enumerate(deltas):
    	axs[0].plot(nsamps, np.mean(type1_err,axis=2)[:,j], '-o', label='$\delta$ = '+delta_string[j])
    axs[0].set_xlabel('N')
    axs[0].set_ylabel('Type 1 Error')
    axs[0].set_xscale('log')
    
    for (j,delta) in enumerate(deltas):
    	axs[1].plot(nsamps, np.mean(type2_err,axis=2)[:,j], '-o', label='$\delta$ = '+delta_string[j])
    axs[1].set_xlabel('N')
    axs[1].set_ylabel('Type 2 Error')
    axs[1].set_xscale('log')
    
    for (j,delta) in enumerate(deltas):
    	axs[2].plot(nsamps, np.mean(succ_prob,axis=2)[:,j], '-o', label='$\delta$ = '+delta_string[j])
    axs[2].set_xlabel('N')
    axs[2].set_ylabel('Success Prob')
    axs[2].set_xscale('log')
    
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol = 6)
    
    plt.savefig('figures/graph_errors_ord'+str(order)+'.pdf')
