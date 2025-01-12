function ydot = NS_1D_RHS(~,y,mu,K,gamma,dx)

if nargin<4
    mu = 1e-3;
    K = 1e-3;
    gamma = 1.4;
    dx=1;
end

R = 8.314;

rho = y(:,1);
rhoU = y(:,2);
E = y(:,3);

rhoU(1) = 0;
rhoU(end) = 0;

rho(1) = rho(2);
E(1) = E(2);

rho(end) = rho(end-1);
E(end) = E(end-1);

U = rhoU./rho;

dU = utils.Grad(U,dx);

rhoU2 = rhoU.*U;

P = (gamma-1)*(E - 0.5*rhoU2);

T = P./(R*rho);

dT = utils.Grad(T,dx); 
dT = utils.Filter(dT); 

dF1 = utils.Grad(rhoU,dx);
dF2 = utils.Grad(rhoU2 + P,dx);
dF3 = utils.Grad((E+P).*U,dx);

dS2 = utils.Grad((4/3)*mu*dU,dx);
dS3 = utils.Grad((4/3)*mu*U.*dU + K*dT,dx);

ydot = [-dF1 dS2-dF2 dS3-dF3];

ydot(1,:) = 0;
ydot(end,:) = 0;

end