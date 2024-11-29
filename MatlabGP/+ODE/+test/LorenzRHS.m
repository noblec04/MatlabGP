function ydot = LorenzRHS(t,y,sigma,beta,rho)

if nargin<3
    sigma = 10;
    beta = 8/3;
    rho = 28;
end

ydot = [
    sigma * (y(2) - y(1));
    y(1) * (rho - y(3)) - y(2);
    y(1) * y(2) - beta * y(3)
    ];
end