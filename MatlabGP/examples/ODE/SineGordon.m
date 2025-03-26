clear
clc

tf = 150;

xs0 = 4; %Shock location
xi0 = 5; %Interface location

ls0 = 1; %Shock width
li0 = 2; %interface width
x = linspace(0,9,200)';

ss = -0*exp(-((x-xs0)/ls0).^2) + 1.5*exp(-((x-xi0)/li0).^2);


dx = 0.5*0.5;

rho = ss;

U = 2*((x-xs0)/ls0).*exp(-((x-xs0)/ls0).^2) - 2*((x-xi0)/li0).*exp(-((x-xi0)/li0).^2);

y0 = [rho rho.*U];


%%

tic
[tf, yf] = ODE.rkf45(@(t,y) ODE.test.SineGordon_RHS(t,y,dx), y0, 0, tf, 0.001,1e-4);
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
%subplot(2,2,1)
pcolor(x,tf,squeeze(yfv(:,1,:))')
shading flat
utils.cmocean('balance')
colorbar

% subplot(2,2,2)
% pcolor(x,tf,squeeze(yfv(:,2,:))'./squeeze(yfv(:,1,:))')
% shading flat
% utils.cmocean('balance')
% colorbar
% 
% subplot(2,2,3)
% pcolor(x,tf,utils.Grad(squeeze(yfv(:,1,:))',dx).^2)
% shading flat
% utils.cmocean('balance')
% colorbar
% caxis([0 0.0001])