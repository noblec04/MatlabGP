clear
clc

gamma = 1.4;
mu = 5e-1;
K = 5e-1;

tf = 10;

xs0 = 2; %Shock location
xi0 = 7; %Interface location

ls0 = 0.003; %Shock width
li0 = 0.003; %interface width

x = linspace(0,9,200)';
y = linspace(-1,1,100)';

[Y,X] = meshgrid(y,x);

ss = 0.5*erfc((X-xs0)/ls0);

ss2 = 3 + (1 - 3)*0.5*erfc((X-xi0 + 0.5*cos(10*Y))/li0);

dx = 0.8;
dy=dx;

rho = 0.125 + (1 - 0.125)*ss;

rho = rho.*ss2;

U = 0*rho;
V = 0*rho;
P = 0.1 + (1.2 - 0.1)*ss;

E = P./(gamma-1) + 0.5*rho.*U.^2 + 0.5*rho.*V.^2;

y0(:,:,1) = rho;
y0(:,:,2) = rho.*U;
y0(:,:,3) = rho.*V;
y0(:,:,4) = E;


dy0 = ODE.test.NS_2D_RHS(0,y0,mu,K,gamma,dx,dy);

%%

tic
[tf, yf] = ODE.TVD_rk3(@(t,y) ODE.test.NS_2D_RHS(t,y,mu,K,gamma,dx,dy), y0, 0, tf, 0.03);
toc

yfv=[];

for i = 1:length(tf)
    yfv = yf{i};
    rv(:,:,i) = squeeze(yfv(:,:,1));
    Uv(:,:,i) = squeeze(yfv(:,:,2))./rv(:,:,i);
    Vv(:,:,i) = squeeze(yfv(:,:,3))./rv(:,:,i);
    Ev(:,:,i) = squeeze(yfv(:,:,4))./rv(:,:,i);
end 

%%
figure
pcolor(x,y,squeeze(rv(:,:,1))')
shading flat
utils.cmocean('thermal')
colorbar
