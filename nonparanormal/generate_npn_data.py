#
# This file generates nonparanormal data
#
import warnings
import time
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import scipy.integrate as integrate

# sample graph
def sample_graph(p):
    omega_0 = np.zeros((p,p)) #inverse correlation matrix, to be filled
    s = 3 # controls level of sparsity
    for i in range(p):
        edge_i = 0
        for j in range(i):
            # sample Y_i^1, Y_i^2, Y_j^1, Y_j^2
            y_i = np.random.rand(2,1)
            y_j = np.random.rand(2,1)
            p_ij = 1/(2*np.pi)*np.exp(-np.linalg.norm(y_i - y_j)**2/2/s)
            alpha = np.random.rand(1)
            if p_ij > alpha:
                if edge_i < 4:
                    omega_0[i][j] = .245
                    omega_0[j][i] = .245
                    edge_i +=1
        omega_0[i][i] = 1
    return omega_0

def sample_npn(omega_0, g_type, n):
    # covariance matrix
    Sigma_0 = np.linalg.inv(omega_0)
    # find p
    p = omega_0.shape[0]
    # set mu_0
    mu_0 = np.zeros((p,)) #np.ones((p,))*.5 
    # sample normal data
    norm_data = np.random.multivariate_normal(mu_0, Sigma_0, n)
    # define empty array for NPN data
    npn_data = np.zeros((n,p))
    
    # define transformation functions
    if g_type == 0:
        # power transformation
        def g_0(t,alpha_j):
            return np.sign(t)*(np.abs(t)**alpha_j)
        def integrand(t, mu_j, sigma_j, alpha_j):
            A = (g_0(t-mu_j, alpha_j))**2
            B = norm.pdf(t, loc=mu_j, scale=sigma_j)
            return A*B
        def g_j(z_j, mu_j, sigma_j, alpha_j):
            A, errA = integrate.quad(integrand, -np.inf, np.inf, args=(mu_j, sigma_j, alpha_j))
            return sigma_j * g_0(z_j - mu_j, alpha_j) / np.sqrt(A) + mu_j
    
    elif g_type == 1:
        # gaussian CDF transformation
        def g_0(t, mu_g0, sigma_g0):
            alpha = 0#-1
            beta = 1
            return (beta - alpha)*norm.cdf(t, loc=mu_g0, scale=sigma_g0) + alpha #map from alpha to beta
        def integrand1(t, mu_g0, sigma_g0, mu_j, sigma_j):
            return norm.cdf(t, loc=mu_g0, scale=sigma_g0) * norm.pdf(t, loc=mu_j, scale=sigma_j)
        def integrand2(y, mu_g0, sigma_g0, mu_j, sigma_j):
            B, errB = integrate.quad(integrand1, -np.inf, +np.inf, args=(mu_g0, sigma_g0, mu_j, sigma_j))
            return (norm.cdf(y, mu_g0, sigma_g0) - B)**2 * norm.pdf(y, mu_j, sigma_j)
        def g_j(z_j, mu_g0, sigma_g0, mu_j, sigma_j):
            A = g_0(z_j, mu_g0, sigma_g0)
            B, errB = integrate.quad(integrand1, -np.inf, +np.inf, args=(mu_g0, sigma_g0, mu_j, sigma_j))
            C, errC = integrate.quad(integrand2, -np.inf, +np.inf, args=(mu_g0, sigma_g0, mu_j, sigma_j))
            return sigma_j * (A - B) / np.sqrt(C) + mu_j

    if g_type == 0:
        alpha = 3.
        for j in range(p):
            mu_j = mu_0[j]
            sigma_j = np.sqrt(Sigma_0[j][j])
            alpha_j = alpha
            npn_data[:,j] = g_j(norm_data[:,j], mu_j, sigma_j, alpha_j)
    
    elif g_type == 1:
        mu_g0 = 0.05
        sigma_g0 = .4
        for j in range(p):
            mu_j = mu_0[j]
            sigma_j = np.sqrt(Sigma_0[j][j])
            npn_data[:,j] = g_j(norm_data[:,j], mu_g0, sigma_g0, mu_j, sigma_j)

    return npn_data
