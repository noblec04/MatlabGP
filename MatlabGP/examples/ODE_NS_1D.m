clear
clc

gamma = 1.4;
mu = 1e-1;
K = 1e-1;

tf = 400;

x = linspace(0,1,200)';

ss = 0.5*erfc(100*(x-0.3));

ss2 = 2 + (1 - 2)*0.5*erfc(10*(x-0.7));

dx = 0.8;

rho = 0.125 + (1 - 0.125)*ss;

rho = rho.*ss2;

U = 0*x;
P = 0.1 + (1.2 - 0.1)*ss;

E = P./(gamma-1) + 0.5*rho.*U.^2;

y0 = [rho rho.*U E];

%%

tic
[tf, yf] = ODE.rkf45(@(t,y) ODE.test.NS_1D_RHS(t,y,mu,K,gamma,dx), y0, 0, tf, 0.003,1e-6);
toc

yfv=[];

for i = 1:length(tf)
    yfv(:,:,i) = yf{i};
end 

%%
figure
subplot(2,2,1)
pcolor(x,tf,squeeze(yfv(:,1,:))')
shading flat
utils.cmocean('thermal')
colorbar

subplot(2,2,2)
pcolor(x,tf,squeeze(yfv(:,2,:))'./squeeze(yfv(:,1,:))')
shading flat
utils.cmocean('balance')
colorbar

subplot(2,2,3)
pcolor(x,tf,squeeze(yfv(:,3,:))')
shading flat
utils.cmocean('thermal')
colorbar

subplot(2,2,4)
pcolor(x,tf,utils.Grad(squeeze(yfv(:,1,:))',dx).^2)
shading flat
utils.cmocean('thermal')
colorbar
caxis([0 0.0001])