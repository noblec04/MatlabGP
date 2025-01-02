# MatlabGP
flexible GP model with user friendly kernel construction inspired by STHENO.

--------------------------------------------------------------------------------------------------------------------------------------------------------

+kernels
  - CEQ 
  - DECAY
  - EPAN
  - EQ - squared exponential
  - EQ_matrix - squared exponential with off diagonal lengthscale terms
  - GE
  - JumpRELU
  - RELU
  - TANH
  - WN - white noise
  - LOG
  - LPEQ
  - RQ - rational quadratic
  - Matern12
  - Matern32
  - Matern52
  - Lin - Linear
  - GE - Gaussian Envelope

+means
  - zero
  - linear
  - sine
  - const

+BO
  - argmax
  - argmaxGrid
  - argmin
  - batchopt
  - EI
  - FUNBO
  - LCB
  - maxGrad
  - maxMU
  - maxVAR
  - MFSFDelta
  - minMU
  - opt
  - UCB

+NN
  - AE
  - BFF
  - C2
  - CE
  - FF
  - LIN
  - MAE
  - MFNN
  - MSE
  - NLL
  - NN
  - RELU
  - reshape
  - SNAKE
  - SWISH
  - TANH



--------------------------------------------------------------------------------------------------------------------------------------------------------

GP - Exact GP with gaussian likelihood

VGP - Variational GP with gaussian likelihood

MFGP - An AR(1) multi-fidelity GP using Le Gratiet simplification (nF Cov matrices rather than 1 large Cov matrix)

NLMFGP - An AR(N) non-linear GP

NN - Neural Network with a number of different layers, architectures and loss functions, all powered by AutoDiff. 
--------------------------------------------------------------------------------------------------------------------------------------------------------

Means can be added or multiplied together or divided. Kernels can be added or multiplied.

GP models find hyperparameters via the mean of the posterior.

--------------------------------------------------------------------------------------------------------------------------------------------------------
Example:

```matlab:Code
clear all
%close all
clc

f1 = @(x) (6*x-2).^2.*sin(12*x-4);

xx = linspace(0,1,100)';
yy = f1(xx);

x1 = [0; 1*lhsdesign(8,1);1];
y1 = f1(x1);
```

```matlab:Code
a = means.const(2) + means.linear(4);

b = (kernels.Matern52(1,0.2) + kernels.EQ(0.2,0.4))*kernels.RQ(2,1,0.1);
b.signn = 0.2;
```

```matlab:Code
Z = GP(a,b);
```

```matlab:Code
Z1 = Z.condition(x1,y1);

figure
utils.plotLineOut(Z1,1,1)
hold on
plot(xx,yy,'-.')
plot(x1,y1,'+','MarkerSize',12,'LineWidth',3)
```

```matlab:Code

for i = 1:30
    ysamp = Z1.samplePosterior(xx);
    plot(xx,ysamp,'LineWidth',0.05,'Color','k')
end

```

```matlab:Code
tic
[Z2] = Z1.train();
toc
```

```text:Output
Elapsed time is 1.141292 seconds.
```

```matlab:Code
figure
hold on
utils.plotLineOut(Z2,1,1)
plot(xx,yy,'-.')
plot(x1,y1,'+','MarkerSize',12,'LineWidth',3)
```

```matlab:Code
figure
hold on
tic
for i = 1:5
    [x{i},R{i}] = BO.argmax(@BO.maxVAR,Z2);
    plot(x{i},R{i},'x','MarkerSize',18,'LineWidth',4)
end
xlim([0 1])
toc

```
