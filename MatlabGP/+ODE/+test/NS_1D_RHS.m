function ydot = NS_1D_RHS(t,y,mu,K,gamma,dx)

if nargin<4
    mu = 1e-3;
    K = 1e-3;
    gamma = 1.4;
    dx=1;
end

nF = 5;
dd = 0.47;

R = 8.314;

rho = y(:,1);
rhoU = y(:,2);
E = y(:,3);

rho = utils.Filter(rho,dd,nF);
rhoU = utils.Filter(rhoU,dd,nF);
E = utils.Filter(E,dd,nF);

U = rhoU./rho;

dU = utils.Grad(U,dx);

rhoU2 = rhoU.*U;

P = (gamma-1)*(E - 0.5*rhoU2);

T = P./(R*rho);

dT = utils.Grad(T,dx); 

dF1 = utils.Grad(rhoU,dx);
dF2 = utils.Grad(rhoU2 + P,dx);
dF3 = utils.Grad((E+P).*U,dx);

dF1 = utils.Filter(dF1,dd,nF);
dF2 = utils.Filter(dF2,dd,nF);
dF3 = utils.Filter(dF3,dd,nF);

dS2 = utils.Grad((4/3)*mu*dU,dx);
dS3 = utils.Grad((4/3)*mu*U.*dU + K*dT,dx);

dS2 = utils.Filter(dS2,dd,nF);
dS3 = utils.Filter(dS3,dd,nF);

ydot = [-dF1 dS2-dF2 dS3-dF3];

ydot(1,:)=0;
ydot(end,:)=0;
 
end