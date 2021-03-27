#
# This file is part of TransportMaps.
#
# TransportMaps is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#e
# TransportMaps is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with TransportMaps.  If not, see <http://www.gnu.org/licenses/>.
#
# Transport Maps Library
# Copyright (C) 2015-2018 Massachusetts Institute of Technology
# Uncertainty Quantification group
# Department of Aeronautics and Astronautics
#
# Authors: Transport Map Team
# E-mail: tmteam@mit.edu
# Website: transport-maps.mit.edu
# Support: transport-maps.mit.edu/qa/
#

import numpy as np
import matplotlib.pyplot as plt
import itertools

#from TransportMaps.Defaults import \
#    Default_IsotropicIntegratedSquaredTriangularTransportMap, \
#    Default_IsotropicIntegratedExponentialTriangularTransportMap
from TransportMaps.Distributions import \
    StandardNormalDistribution, DistributionFromSamples, \
    PushForwardTransportMapDistribution, PullBackTransportMapDistribution
import GeneralizedPrecision as GP
from TransportMaps import Default_IsotropicIntegratedSquaredTriangularTransportMap

__all__ = ['SING']

# inputs: data (n x d), polynomial order, node ordering method, tolerance delta
# output: sparse edge set
def SING(data, p_order, ordering, delta, offset=0, REG=None, plotting=False, delta_str=None):
    r""" Identify a sparse edge set of the undirected graph corresponding to the data.

    Args: 
      data: data :math:`{\bf x}_i, i = 1,\dots,n`,
      p_order (int): polynomial order of the transport map representation,
      ordering: scheme used to reorder the variables,
      delta (float): tolerance :math:`\delta`
      REG (dict): regularization dictionary with keys 'type' and 'reweight'

    Returns:
      the generalized precision :math:`\Omega`
    """
    # Print inputs to user
    print("Problem inputs:\n\n dimension = ",data.shape[1],
          "\n number of samples = ",data.shape[0],
          "\n polynomial order = ",p_order,
          "\n ordering type = ", ordering.__class__,
          "\n delta = ",delta,"\n")
    # Initial setup
    # data is (n x d)
    dim = data.shape[1]
    n_samps = data.shape[0]
    nax = np.newaxis
    pi = {}

    # Target density is standard normal
    eta = StandardNormalDistribution(dim)
    # Quadrature type and params
    # 0: Monte Carlo quadrature with n point
    qtype = 0
    qparams = n_samps
    # Gradient information
    # 0: derivative free
    # 1: gradient information
    # 2: Hessian information
    ders = 2
    # Tolerance in optimization
    tol = 1e-5
    # Log stores information from optimization
    log = [ ]

    # All variables are active variables to begin
    active_vars=None
    # Set initial sparsity level
    sparsity = [0]
    sparsityIncreasing = True
    # Number of active variables
    n_active_vars = [(np.power(dim,2) + dim)/2]
    # Create list to store permutations
    perm_list = []
    # Create total_perm vector
    total_perm = np.arange(dim)
    # create counter for iterations
    counter = 0
    while sparsityIncreasing:
    
       # Define base_density from samples
       pi = DistributionFromSamples(data)

       # Build the transport map (isotropic for each entry)
       tm_approx = Default_IsotropicIntegratedSquaredTriangularTransportMap(
           dim, p_order, active_vars=active_vars, btype='fun')

       # Construct density T_\sharp \pi
       tm_density = PushForwardTransportMapDistribution(tm_approx, pi)

       # Set regularization in REG
       REG = {'type': None }
       solve = tm_density.minimize_kl_divergence(eta, qtype=qtype,
                                                 qparams=qparams,
                                                 regularization=REG,
                                                 tol=tol, ders=ders)
       
       pb_density = PullBackTransportMapDistribution(tm_approx, eta)

       # Compute generalized precision
       omegaHat = GP.gen_precision(pb_density, data)
       
       # Compute variance of generalized precision
       gp_var = GP.var_omega(pb_density, data)
       # Compute tolerance (matrix)
       tau = delta * np.sqrt(gp_var) * np.sqrt(np.log(n_samps)) # set tolerance as square root of variance

       # Save diagonal elements
       omegaHat_diagonal = np.copy(np.diag(omegaHat))
       # Threshold omegaHat
       omegaHat[np.abs(omegaHat) < offset+tau] = 0
       # Put diagonal elements back in
       omegaHat[np.diag_indices_from(omegaHat)] = omegaHat_diagonal
       
       # Reorder variables and data
       perm1 = ordering.perm(omegaHat)
       # Apply to omegaHat
       omegaHat_temp = omegaHat[:,perm1][perm1,:]
       # Check if ordering would flip if applied again (problem for reverse Cholesky)
       perm2 = ordering.perm(omegaHat_temp)

       if (perm2 == perm1).all():
       #if (perm2 == perm1):
           perm_vect = np.arange(dim) # set to identity permutation
       else:
           perm_vect = perm1
       
       # Apply to omegaHat
       omegaHat = omegaHat[:,perm_vect][perm_vect,:]
       # Apply re-ordering to data
       data = data[:, perm_vect]
    
       # Add permutation to perm_list
       perm_list.append(perm_vect)
       # Convolve permutation with total_perm
       total_perm = total_perm[perm_vect]
       inverse_perm = [0] * dim
       for i, p in enumerate(total_perm):
           inverse_perm[p] = i
    
       # Extract lower triangular matrix
       omegaHatLower = np.tril(omegaHat)
       # Count edges
       edge_count = np.count_nonzero(omegaHatLower) - dim
    
       # Variable elimination moving from highest node (dim-1) to node 2 (at most)
       for i in range(dim-1,1,-1):
           non_zero_ind  = np.where(omegaHatLower[i,:i] != 0)[0]
           if len(non_zero_ind) > 1:
               co_parents = list(itertools.combinations(non_zero_ind,2))
               for j in range(len(co_parents)):
                   row_index = max(co_parents[j])
                   col_index = min(co_parents[j])
                   omegaHatLower[row_index, col_index] = 1.0

       # Find list of active_vars
       active_vars = []
       for i in range(dim):
           actives = np.where(omegaHatLower[i,:] != 0)
           active_list = list(set(actives[0]) | set([i]))
           active_list.sort(key=int)
           active_vars.append(active_list)
       
       # Find n_active_vars
       n_active_vars.append(np.sum([len(x) for x in active_vars]))
    
       # Find current sparsity level
       sparsity.append(n_active_vars[0] - n_active_vars[-1])
    
       # Set sparsityIncreasing
       if sparsity[-1] <= sparsity[-2]:
           sparsityIncreasing = False

       # Print statement for SING
       counter = counter + 1
       print('\nSING Iteration: ', counter)
       print('Active variables: ', active_vars, '\n  Note variables may be permuted.')
       print('Number of edges: ', edge_count,' out of ', np.int((dim**2 - dim)/2),' possible')

       # print Omega iterations if specified
       if plotting == True:
           omegaHat_plot = omegaHat[:,inverse_perm][inverse_perm,:]
           if delta_str == None:
               delta_str = str(delta)
           np.savetxt('./gp_iter/graph_order_' + str(p_order) + '_N' + str(data.shape[0]) + \
            '_delta' + delta_str + '_iter' + str(counter) + '.txt', omegaHat_plot, fmt='%1.3f')

    # Recovered omega (same variable order as input ordering)
    rec_omega = omegaHat[:,inverse_perm][inverse_perm,:]
    
    print('\nSING has terminated.')
    print('Total permutation used: ',total_perm)
    graph = np.zeros((dim,dim))
    graph[np.nonzero(rec_omega)] = 1
    print('Recovered graph: \n', graph)
    return rec_omega
