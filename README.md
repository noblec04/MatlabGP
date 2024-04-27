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

--------------------------------------------------------------------------------------------------------------------------------------------------------
Example:

f1 = @(x) (6*x-2).^2.*sin(12*x-4);

x1 = lhsdesign(6,1);

y1 = f1(x1);

a = means.linear(4)*means.sine(1,3,0,1);

b = (kernels.Matern52(1,0.2) + kernels.EQ(0.2,0.4))*kernels.RQ(1,0.1);

b.signn = 0; %Nugget term

Z = GP(a,b);

Z1 = Z.condition(x1,y1);

Z2 = Z1.train();

[ys,sig] = Z2.eval(xx);

figure

plot(xx,ys)

hold on

plot(xx,ys+2*sqrt(sig),'--')

plot(xx,ys-2*sqrt(sig),'--')

plot(xx,yy,'-.')

plot(x1,y1,'+')
