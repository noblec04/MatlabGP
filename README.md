# MatlabGP
flexible GP model with user friendly kernel construction inspired by STHENO.

+kernels
  - EQ - squared exponential
  - RQ - rational quadratic
  - Matern52
  - Lin - Linear
  - GE - Gaussian Envelope

+means
  - zero
  - linear
  - sine
  - const

GP - Exact GP with gaussian likelihood
VGP - Variational GP with gaussian likelihood
MFGP - An AR(1) multi-fidelity GP using Le Gratiet simplification (nF Cov matrices rather than 1 large Cov matrix)

Means can be added or multiplied together or divided. Kernels can be added or multiplied.

GP and MFGP models find hyperparameters via the mean of the posterior.

VGP finds the hyperparameters as MAP point values. This uses the BADS (bayesian adaptive direct search) package.

WIP:
 -Analytical gradients for all hyperparameters. All HPs will be found using VSGD (variational stochastic gradient decent) if number of HPs is large.
