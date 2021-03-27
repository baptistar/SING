import numpy as np
from scipy.stats import norm
from scipy.stats import bernoulli
import itertools
import warnings
import h5py

class Rademacher():
    def __init__(self,dim):
        if dim%2 != 0:
            raise ValueError('Input Dimension Must be an Even Number')
        self.dim = dim

    def edgeSet(self):
        omega = np.eye(self.dim)
        for i in range(0,self.dim,2):
            omega[i,i+1] = 1
            omega[i+1,i] = 1
        return omega

    def sigma(self):
        sigma = np.eye(self.dim)
        return sigma

    def rvs(self, N):
        rvs = np.zeros([N, self.dim])
        for i in range(0,self.dim,2):
            rvs[:,i] = norm.rvs(size=N)
            rade = 2*bernoulli.rvs(p=.5, size=N) - 1
            rvs[:,i+1] = rvs[:,i] * rade
            print(rvs.shape[1])
        return rvs

class Butterfly():
    def __init__(self,dim):
        if dim%2 != 0:
            raise ValueError('Input Dimension Must be an Even Number')
        self.dim = dim

    def edgeSet(self):
        omega = np.eye(self.dim)
        for i in range(0,self.dim,2):
            omega[i,i+1] = 1
            omega[i+1,i] = 1
        return omega

    def sigma(self):
        sigma = np.eye(self.dim)
        return sigma

    def rvs(self, N):
        rvs = np.zeros([N, self.dim])
        for i in range(0,self.dim,2):
            rvs[:,i]   = norm.rvs(size=N)
            gaussian_i = norm.rvs(size=N)
            rvs[:,i+1] = rvs[:,i] * gaussian_i
        return rvs


class Uniform():
    def __init__(self,dim):
        if dim%2 != 0:
            raise ValueError('Input Dimension Must be an Even Number')
        self.dim = dim

# ***************this is not really omega, but adjacency matrix!***********************
    def omega(self):
        omega = np.eye(self.dim)
        for i in range(0,self.dim,2):
            omega[i,i+1] = 1
            omega[i+1,i] = 1
        return omega

    def sigma(self):
        sigma = np.eye(self.dim)
        return sigma

    def rvs(self, N):
        rvs = np.zeros([N, self.dim])
        for i in range(0,self.dim,2):
            rvs[:,i] = norm.rvs(size=N)
            unif = norm.rvs(size=N)
            rvs[:,i+1] = rvs[:,i] * unif
            print(rvs.shape[1])
        return rvs

    def n_edges(self):
        return self.dim/2.


class StochVolHyper():
    def __init__(self, nsamps):
        if nsamps>1e5:
            raise ValueError('Number of samples must be less than 1e5')
        self.nsamps = nsamps

# ***************this is not really omega, but adjacency matrix!***********************
    def omega(self):
        # create tridiagonal matrix 
        omega = np.eye(self.dim) + np.eye(self.dim,k=-1) + np.eye(self.dim,k=+1)
        omega[:,0:2] = np.ones((self.dim+3,3))
        omega[0:2,:] = np.ones((3,self.dim+3))
        omega[0,2] = 0
        omega[1,2] = 0
        omega[2,0] = 0
        omega[2,1] = 0
        return omega

    def rvs(self, n):
        f = h5py.File('~/Dropbox/Sparsity/StochasticVolatility/SVHDurbin-n10-phi095-sigma025-mu-05-dir-comp-n10000-veps1e-3-postprocess-analysis.dill.hdf5','r')
        samples = f['samples']
        rvs = samples[:n,:]
        return rvs

    def n_edges(self):
        n_edges = (self.dim - 1) + 1 + self.dim*3
        return n_edges

class StarGraph():
    def __init__(self,dim):
        self.dim = dim

    def omega(self):
        # create star matrix 
        omega = np.eye(self.dim)
        omega[0,0] = self.dim
        omega[0,1:] = .3
        omega[1:,0] = .3
        # compute unnormalized sigma (covariance matrix)
        sigma_temp = np.linalg.inv(omega)
        # extract standard deviations from sigma_temp
        std_temp = np.diag(np.sqrt(np.diag(sigma_temp)))
        # scale omega by std_temp
        omega = np.dot(np.dot(std_temp, omega), std_temp)
        return omega

    def sigma(self):
        # compute sigma as inverse of omega
        sigma = np.linalg.inv(self.omega())
        return sigma
        
    def rvs(self, N):
        # compute K, where K K^T = Sigma
        K = np.linalg.cholesky(self.sigma())
        # generate 'base' samples from standard normal
        mu = np.zeros((self.dim,))
        cov = np.eye(self.dim)
        x = np.random.multivariate_normal(mu,cov,size=N)
        # return N samples from distribution
        samples = np.dot(K,x.T).T
        return samples

    def active_vars(self):

        # find zero elements in omega to determine active_vars
        omegaLower = np.tril(self.omega())
        active_vars = []
        for i in range(self.dim):
            actives = np.where(omegaLower[i,:] != 0)
            active_list = list(set(actives[0]) | set([i]))
            active_list.sort(key=int)
            active_vars.append(active_list)

        return active_vars

class GridGraph():
    def __init__(self,dim):
        if (np.sqrt(dim) - int(np.sqrt(dim))) != 0:
            raise ValueError('Input Dimension Must be a Square Number')
        self.dim = dim

    def omega(self):

        dim_sq_root = int(np.sqrt(self.dim))

        # declare grid coordinates
        coords = self.zigzag(dim_sq_root)
        n_coords = len(coords)
    
        # create zero matrix
        omega = np.zeros((n_coords, n_coords))

        # pull out all coordinates
        all_coords = list(coords.values())

        # add all edges for the grid graph
        for i in range(n_coords):
            coord_val = coords[i];
            new_coords = [(coord_val[0],coord_val[1]+1), 
                          (coord_val[0],coord_val[1]-1), 
                          (coord_val[0]+1,coord_val[1]),
                          (coord_val[0]-1,coord_val[1])]
            for j in range(len(new_coords)):
                if new_coords[j] in all_coords:
                    coord_idx = all_coords.index(new_coords[j])
                    omega[i, coord_idx] = 1.0
                    omega[coord_idx, i] = 1.0

        # set the diagonal appropriately
        max_val = np.ceil(np.max(np.sum(np.abs(omega), axis = 0))) + 1
        omega = omega + max_val*np.eye(n_coords)
        
        # compute unnormalized sigma (covariance matrix)
        sigma_temp = np.linalg.inv(omega)
        # extract standard deviations from sigma_temp
        std_temp = np.diag(np.sqrt(np.diag(sigma_temp)))
        # scale omega by std_temp
        omega = np.dot(np.dot(std_temp, omega), std_temp)
        return omega

    def zigzag(self,n):
        # zig-zag pattern returns bijection between graph coordinates and ordering
        indexorder = sorted(((x,y) for x in range(n) for y in range(n)),
                    key = lambda p: (p[0]+p[1], -p[1] if (p[0]+p[1]) % 2 else p[1]) )
        return dict((n,index) for n,index in enumerate(indexorder))


    def sigma(self):
        # compute sigma as inverse of omega
        sigma = np.linalg.inv(self.omega())
        return sigma
        
    def rvs(self, N):
        #dim_sq = np.power(self.dim,2)
        # compute K, where K K^T = Sigma
        K = np.linalg.cholesky(self.sigma())
        # generate 'base' samples from standard normal
        mu = np.zeros((self.dim,))
        cov = np.eye(self.dim)
        x = np.random.multivariate_normal(mu,cov,size=N)
        # return N samples from distribution
        samples = np.dot(K,x.T).T
        return samples

    def active_vars(self):
        #dim_sq = np.power(self.dim,2)

        # extract lower triangular matrix
        omegaLower = np.tril(self.omega())

        # add edges by...
        # variable elimination moving from highest node (dim-1) to node 2 (at most)
        for i in range(self.dim-1,1,-1):
            non_zero_ind  = np.where(omegaLower[i,:i] != 0)[0]
            if len(non_zero_ind) > 1:
                co_parents = list(itertools.combinations(non_zero_ind,2))
                for j in range(len(co_parents)):
                    row_index = max(co_parents[j])
                    col_index = min(co_parents[j])
                    omegaLower[row_index, col_index] = 1.0

        # find zero elements in chordal omega to determine active_vars
        active_vars = []
        for i in range(self.dim):
            actives = np.where(omegaLower[i,:] != 0)
            active_list = list(set(actives[0]) | set([i]))
            active_list.sort(key=int)
            active_vars.append(active_list)

        return active_vars
