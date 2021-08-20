import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import SpectralToolbox.Spectral1D as S1D

import TransportMaps.Distributions as DIST
import TransportMaps.Functionals as FUNC

# Initialize parameters
npair = 5
num_sample = 100
mod_rad_data = np.zeros((num_sample, npair*2))
X = np.random.randn(num_sample, npair)
W = np.random.randn(num_sample, npair)
Y = X*W
mod_rad_data = np.hstack((X, Y))

# Class that defines and optimizes a transport map component
class TransportMapComponent():
    def __init__(self, data, u, v, S, order):
        # Extract sub dataset
        self.data = data
        self.u = u
        self.v = v
        self.S = S
        self.extractdata = data[:, [u, v] + S]
        # Parameters of the sub dataset
        self.numsample = data.shape[0]
        self.dim = len(S) + 2
        # Order of polynomials in the parametrization
        self.order = order

    def compute_proxy(self):
        # Define the list of basis and polynomial orders
        c_basis_list = [S1D.HermiteProbabilistsPolynomial()]*self.dim
        c_orders_list = [self.order]*(self.dim - 1) + [0]
        h_basis_list = [S1D.HermiteProbabilistsPolynomial()]*(self.dim-1) + \
                       [S1D.ConstantExtendedHermiteProbabilistsFunction()]
        h_orders_list = [self.order]*self.dim
        # Define the linear approximations
        c_approx = FUNC.MonotonicLinearSpanApproximation(
            c_basis_list, spantype='total', order_list=c_orders_list)
        h_approx = FUNC.MonotonicLinearSpanApproximation(
            h_basis_list, spantype='total', order_list=h_orders_list)

        # print(self.extractdata)

        # Define the Integrated Exponential parametrization
        Tk = FUNC.MonotonicIntegratedExponentialApproximation(c_approx, h_approx)
        # Define the target scalar normal distribution
        pi_dist = DIST.StandardNormalDistribution(1)
        # Define the loss function induced from the pushforward density
        f = FUNC.ProductDistributionParametricPullbackComponentFunction(Tk, pi_dist)

        # Define a map component for the optimization problem
        map_component = FUNC.MonotonicIntegratedExponentialApproximation(c_approx, h_approx)
        # Define the quadrature (equal weights)
        weight = np.array([1/self.numsample]*self.numsample)
        # Find the optimal coefficients that minimize the loss function f
        optimal_coeff = map_component.minimize_kl_divergence_component(f, self.extractdata, weight)
        f.coeffs = optimal_coeff['jac']

        # Compute the corresponding proxy
        hessian_scores = f.hess_x(self.extractdata)
        proxy = np.mean(hessian_scores[:,[0],[1]]**2)

        return proxy


def neighborhood_selection_proxy(data, u, epsilon, order):
    # Prepare node lists
    dim = data.shape[1]
    node_list = list(set(range(dim)) - set([u]))
    neighbors = []
    maximum = 1

    # selection step
    while maximum > epsilon and len(node_list) > 0:
        # Check how the pseudo neighborhood set varies (the algorithm's efficiency)
        print(neighbors)
        print(maximum)

        # Compute values of proxy
        score_list = [0 for i in range(len(node_list))]
        for i in range(len(node_list)):
            node = node_list[i]

            Smap1 = TransportMapComponent(data, u, node, neighbors, order)
            score_list[i] = Smap1.compute_proxy()

        # Adjust current set of nodes
        print(score_list)
        maximum = max(score_list)
        node_i = node_list[np.argmax(score_list)]
        neighbors += [node_i]
        node_list = list(set(node_list) - set([node_i]))

    # pruning step
    for node_j in neighbors:
        neigh_j = list(set(neighbors) - set([node_j]))
        Smap2 = TransportMapComponent(data, u, node_j, neigh_j, order)

        if Smap2.compute_proxy() < epsilon:
            neighbors = list(set(neighbors) - set([node_j]))

    return neighbors

# Define testing node
utest = 2
epsilon = 1e-10
poly_order = 3
neighbor_list_proxy = neighborhood_selection_proxy(mod_rad_data, utest, epsilon, poly_order)
print(neighbor_list_proxy)
