function ydot = LorenzRHS(t,y,sigma,beta,rho)

if nargin<3
    sigma = 10;
    beta = 8/3;
    rho = 28;
end

y1 = y(1);
y2 = y(2);
y3 = y(3);

ydot = [
    sigma * (y2 - y1);
    y1 * (rho - y3) - y2;
    y1 * y2 - beta * y3
    ];
end