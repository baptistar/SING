'''This file includes multiple ways to reorder the graph nodes, including reverse cholesky, min-fill, and min-degree.'''
import numpy as np
import copy

from sksparse.cholmod import cholesky
from scipy.sparse import csc_matrix
import itertools

__all__ = ['ReverseCholesky', 'MinFill', 'MinDegree']

nax = np.newaxis
class ReverseCholesky():
    def __init__(self):
        self.id = 0

    def perm(self, gp):
        dim = gp.shape[0]
        chol_factor = cholesky(csc_matrix(gp))
        perm = chol_factor.P()
        # reverse ordering
        perm = np.flipud(perm)
        # # inverse perm
        # inv_perm = [0] * dim
        # for i, p in enumerate(perm):
        #     ip[p] = i
        return perm

    def invPerm(self, gp):
        dim = gp.shape[0]
        # inverse perm
        perm = self.perm(gp)
        invPerm = [0] * dim
        for i, p in enumerate(perm):
            invPerm[p] = i
        return np.array(invPerm)

class MinFill():
    def __init__(self):
        self.id = 1

    def perm(self, gp):
        dim = gp.shape[0]
        gp_copy = copy.copy(gp)
        perm = np.zeros(dim,dtype='int')
        nodes_avail = [i for i in range(dim)]
        for k in range(dim):
            # non_zero_sum = np.zeros(dim);
            added_edges = np.zeros(dim);
            for i in range(dim):
                if i in nodes_avail:
                    # non_zero_sum[i] = np.sum(gp_copy[i,:] != 0)
                    gpLower = np.tril(gp_copy)
                    non_zero_ind  = np.where(gpLower[k,:k] != 0)[0]
                    co_parents = list(itertools.combinations(non_zero_ind,2))
                    for j in range(len(co_parents)):
                        row_index = max(co_parents[j])
                        col_index = min(co_parents[j])
                        if gp_copy[row_index, col_index] == 0:
                            added_edges[i] += 1
                else: 
                    added_edges[i] = dim + 1 # make sure not the minimum
            m, ind = min((s,ind) for ind,s in enumerate(added_edges))
            # set value of permutation vector
            perm[k] = ind
            nodes_avail.remove(ind)
            # if no fill, just set value
            if m == 0:
                pass#print("ok!")
            # else, must fill
            else:
                #marry parents of removed edge
                gpLower = np.tril(gp_copy)
                non_zero_ind  = np.where(gpLower[ind,:ind] != 0)[0]
                co_parents = list(itertools.combinations(non_zero_ind,2))
                for j in range(len(co_parents)):
                    row_index = max(co_parents[j])
                    col_index = min(co_parents[j])
                    gp_copy[row_index, col_index] = 1.0
                    gp_copy[col_index, row_index] = 1.0

        perm = np.flipud(perm)
        perm = np.ndarray.tolist(perm)

        return np.array(perm)

    def invPerm(self, gp):
        dim = gp.shape[0]
        # inverse perm
        perm = self.perm(gp)
        invPerm = [0] * dim
        for i, p in enumerate(perm):
            invPerm[p] = i
        return np.array(invPerm)

class MinDegree():
    def __init__(self):
        self.id = 2

    def perm(self, gp):
        dim = gp.shape[0]
        gp_copy = copy.copy(gp)
        perm = np.zeros(dim,dtype='int')
        nodes_avail = [i for i in range(dim)]
        for k in range(dim):
            non_zero_sum = np.zeros(dim);
            for i in range(dim):
                if i in nodes_avail:
                    non_zero_sum[i] = np.sum(gp_copy[i,:] != 0)
                else: 
                    non_zero_sum[i] = dim + 1 # make sure not the minimum
            m, ind = min((s,ind) for ind,s in enumerate(non_zero_sum))
            #set value of permutation vector
            perm[k] = ind
            #marry parents of removed edge
            gpLower = np.tril(gp_copy)
            non_zero_ind  = np.where(gpLower[k,:k] != 0)[0]
            if len(non_zero_ind) > 1:
                co_parents = list(itertools.combinations(non_zero_ind,2))
                for j in range(len(co_parents)):
                    row_index = max(co_parents[j])
                    col_index = min(co_parents[j])
                    gp_copy[row_index, col_index] = 1.0
                    gp_copy[col_index, row_index] = 1.0
            nodes_avail.remove(ind)

        perm = np.flipud(perm)
        perm = np.ndarray.tolist(perm)

        return np.array(perm)

    def invPerm(self, gp):
        dim = gp.shape[0]
        # inverse perm
        perm = self.perm(gp)
        invPerm = [0] * dim
        for i, p in enumerate(perm):
            invPerm[p] = i
        return np.array(invPerm)
