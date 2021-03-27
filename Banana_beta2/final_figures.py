# post process the mc study data of the sparse maps from samples

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import matplotlib
from matplotlib import rc
from math import *
from os import path

rc('text',usetex=True)
font={'family' : 'normal',
    'weight' : 'normal',
    'size' :16}
matplotlib.rc('font',**font)

# define parameters
#nsamps       = np.floor(np.logspace(2,5,num=20)) 
nsamps       = [10000]
deltas	     = [1.] #[np.sqrt(2.)]
delta_string = ['1'] #['sqrt2']
name_string  = 'banana'
dim          = 5
order_vect   = [1,2]
MC_runs      = 1#25

# extract true precision
graph_true = np.eye(dim)
graph_true[0:dim,0] = np.ones((1,dim))
graph_true[0,1:dim] = np.ones((1,dim-1))

for (l, order) in enumerate(order_vect):
    for (j, delta) in enumerate(deltas):

        # define arrays to store errors and success probability
        type1_err = np.zeros((len(nsamps),MC_runs))
        type2_err = np.zeros((len(nsamps),MC_runs))
        succ_prob = np.zeros((len(nsamps),MC_runs))
        type1_err.fill(np.nan)        
        type2_err.fill(np.nan)
        succ_prob.fill(np.nan) 

        # load data from files
        for (i,N) in enumerate(nsamps):
            for k in range(MC_runs):
        
                # load data from file    
                datafile = './outputMC/ban_adj-%s-%d-%d-%s-%d-%d.txt' %(name_string,N,dim,delta_string[j],order,k)
                if (path.exists(datafile) == False):
                    continue
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
        ax.set_xlim((100,1e5))
        ax.set_ylim((-0.5,6.5))
        ax.set_xscale('log')
        #ax.legend(loc='upper right')
        ax.legend(loc='upper left')
        plt.savefig('figures/'+name_string+'_errors_ord'+str(order)+'_delta'+delta_string[j]+'.pdf')
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.errorbar(nsamps, np.mean(succ_prob,axis=1), yerr=1.96*np.std(succ_prob,axis=1)/np.sqrt(MC_runs), fmt='-o', color='g')
        ax.set_xlabel('Number of samples, N')
        ax.set_ylabel('Success Probability')
        ax.set_xlim((100,1e5))
        ax.set_xscale('log')
        plt.savefig('figures/'+name_string+'_succprob_ord'+str(order)+'_delta'+delta_string[j]+'.pdf')
