function ydot = NS_2D_RHS(~,y,mu,K,gamma,dx,dy)

if nargin<3
    mu = 1e-3;
    K = 1e-3;
    gamma = 1.4;
    dx=1;
    dy=1;
end

R = 8.314;

rho = y(:,:,1);
rhoU = y(:,:,2);
rhoV = y(:,:,3);
E = y(:,:,4);

rhoU(1,:) = 0;
rhoU(end,:) = 0;

rhoV(1,:) = 0;
rhoV(end,:) = 0;

rho(1,:) = rho(2,:);
E(1,:) = E(2,:);

rho(end,:) = rho(end-1,:);
E(end,:) = E(end-1,:);

U = rhoU./rho;
V = rhoV./rho;

[dUy,dUx] = utils.Grad(U,dx,dy);
[dVy,dVx] = utils.Grad(V,dx,dy);

rhoU2 = rhoU.*U;
rhoV2 = rhoV.*V;
rhoUV = rhoU.*V;

P = (gamma-1)*(E - 0.5*(rhoU2+rhoV2));

T = P./(R*rho);

[dTy,dTx] = utils.Grad(T,dx,dy); 

dF1 = utils.Grad(rhoU,dx,dy);
dF2 = utils.Grad(rhoU2 + P,dx);
dF3 = utils.Grad(rhoUV,dx);
dF4 = utils.Grad((E+P).*U,dx);

[~,dG1] = utils.Grad(rhoV,dx,dy);
[~,dG2] = utils.Grad(rhoUV,dx,dy);
[~,dG3] = utils.Grad(rhoV2 + P,dx,dy);
[~,dG4] = utils.Grad((E+P).*V,dx,dy);

dS2x = utils.Grad((4/3)*mu*dUx,dx,dy);
[~,dS2y] = utils.Grad((4/3)*mu*dUy,dx,dy);

dS3x = utils.Grad((4/3)*mu*dVx,dx,dy);
[~,dS3y] = utils.Grad((4/3)*mu*dVy,dx,dy);

dS4x = utils.Grad((4/3)*mu*U.*dUx + K*dTx,dx,dy);
[~,dS4y] = utils.Grad((4/3)*mu*V.*dVy + K*dTy,dx,dy);

ydot(:,:,1) = -dF1 - dG1;
ydot(:,:,2) = dS2x + dS2y -dF2 - dG2;
ydot(:,:,3) = dS3x + dS3y -dF3 - dG3;
ydot(:,:,4) = dS4x + dS4y -dF4 - dG4;


ydot(1,:,:) = 0;
ydot(end,:,:) = 0;

end