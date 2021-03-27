# Sparsity Identification in Non-Gaussian Graphical Models (SING) 

## What is the SING Algorithm?

The SING algorithm learns the structure of non-Gaussian graphical models by estimating a conditional independence score matrix. This matrix is based on Hessian information of the joint log-density of a collection of random variables. Given samples from a target density, the algorithm learns a transport map to represent the density and to define a sparse estimator for the score matrix. More information on the algorithm can be found in the [preprint](https://arxiv.org/abs/2101.03093).

## Authors

Ricardo Baptista (MIT), Rebecca Morrison (UC Boulder), Youssef Marzouk (MIT), and Olivier Zahm (INRIA)

E-mails: <rsb@mit.edu>, <rebeccam@colorado.edu>, <ymarz@mit.edu> and <olivier.zahm@inria.fr>

## Installation

The SING algorithm is implemented in Python using the [TransportMaps](https://transportmaps.mit.edu/docs/) package. The algorithm can be found in the `SING` folder. Scripts to reproduce all of the results for each of the examples in the preprint can be found in the remaining folders.

