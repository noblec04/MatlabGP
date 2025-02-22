clear
clc

gamma = 1.4;
mu = 5e-1;
K = 5e-1;

tf = 150;

xs0 = 2; %Shock location
xi0 = 7; %Interface location

ls0 = 0.0001; %Shock width
li0 = 0.1; %interface width
x = linspace(0,9,200)';

ss = 0.5*erfc((x-xs0)/ls0);

ss2 = 3 + (1 - 3)*0.5*erfc((x-xi0)/li0);

dx = 1*0.5;

rho = 0.125 + (1.3 - 0.125)*ss;

rho = rho.*ss2;

U = 0*rho;
P = 0.1 + (1.3 - 0.1)*ss;

E = P./(gamma-1) + 0.5*rho.*U.^2;

y0 = [rho rho.*U E];


%%

tic
[tf, yf] = ODE.rkf45(@(t,y) ODE.test.NS_1D_RHS(t,y,mu,K,gamma,dx), y0, 0, tf, 0.001,1e-9);
toc

%%

yfv=[];
dyfv = [];

for i = 1:length(tf)
    yfv(:,:,i) = yf{i};
    %yfv(:,:,i) = getvalue(yf{i});
    %dyfv(:,:,:,i) = reshape(full(getderivs(yf{i})),[size(squeeze(yfv(:,:,i))) 2]);
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

%%
% figure
% pcolor(x,tf,squeeze(dyfv(:,3,2,:))')
% shading flat
% utils.cmocean('balance')
% colorbar