function ydot = SineGordon_RHS(~,y,dx)

if nargin<3
    dx=1;
end

R = 8.314;

phi = y(:,1);
X = y(:,2);

phi(1) = phi(2);
X(1) = X(2);

phi(end) = phi(end-1);
X(end) = X(end-1);

dphi = utils.Grad(phi,dx); 
d2phi = utils.Grad(dphi,dx); 

dF1 = X;
dF2 = d2phi - sin(phi);

ydot = [dF1 dF2];

ydot(1,:) = 0;
ydot(end,:) = 0;

end