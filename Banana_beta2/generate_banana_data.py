#
# This file generates nonparanormal data
#
import warnings
import time
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import scipy.integrate as integrate

# set number of variables
p = 5

# define banana transformation
def banana_map(X):

    # set parameters
    a = 1.
    b = 1.

    # determine d
    N, d = X.shape
    T = np.zeros((N,d))

    # evaluate map T(X)
    T[:,0] = a*X[:,0];
    for i in range(1,d):
        T[:,i] = -1*b*((a*X[:,0])**2 + a**2) + X[:,i]/a

    # return T(X)
    return T

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

# sample from banana
def sample_npn_beta2(n):

    # sample from banana
    X = np.random.multivariate_normal(np.zeros(p), np.eye(p), n)
    data = banana_map(X)
    
    # transform data with npn
    Sigma_0 = np.cov(data.T)
    mu_0 = np.zeros((p,))
    npn_data = np.zeros((n,p))
    
    mu_g0 = 0.05
    sigma_g0 = .4
    for j in range(p):
        mu_j = mu_0[j]
        sigma_j = np.sqrt(Sigma_0[j][j])
        npn_data[:,j] = g_j(data[:,j], mu_g0, sigma_g0, mu_j, sigma_j)
    return npn_data

