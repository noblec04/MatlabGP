function ydot = BushfireRHS(~,y,alpha,beta,gamma,zeta,eta,nu,f0)

%{

BUSHFIRES AND BALANCE:
PROACTIVE VERSUS REACTIVE POLICIES
IN PRESCRIBED BURNING
SERENA DIPIERRO, ENRICO VALDINOCI, GLEN WHEELER,
AND VALENTINA-MIRA WHEELER

%}

if nargin<3
    alpha = 0.8;
    beta = 0.5;
    gamma = 0.5;
    zeta = 0.14;
    eta = 0.3;
    nu = 0.1;
    f0 = 0.1;
end

y1 = y(1);
y2 = y(2);
y3 = y(3);

ydot = [-1*alpha*y2.*y1-1*beta*y3+gamma*(1-y1); zeta*y1-eta*y2; -1*nu*(y2 - f0).*y1];

end