clear
clc

gamma = 1.4;
mu = 1e-2;
K = 1e-3;

x = linspace(0,1,100)';

ss = 0.5*erfc(20*(x-0.3));

dx = 1;

rho = 0.125 + (1 - 0.125)*ss;
U = 0*x;
P = 0.1 + (1 - 0.1)*ss;

E = P./(gamma-1) + 0.5*rho.*U.^2;

y0 = [rho rho.*U E];

%%

tic
[tf, yf] = ODE.TVD_rk3(@(t,y) ODE.test.NS_1D_RHS(t,y,mu,K,gamma,dx), y0, 0, 30, 0.01);
toc

yfv=[];

for i = 1:length(tf)
    yfv(:,:,i) = yf{i};
end 

%%
figure
subplot(3,1,1)
pcolor(x,tf,squeeze(yfv(:,1,:))')
shading flat
subplot(3,1,2)
pcolor(x,tf,squeeze(yfv(:,2,:))'./squeeze(yfv(:,1,:))')
shading flat
subplot(3,1,3)
pcolor(x,tf,squeeze(yfv(:,3,:))')
shading flat