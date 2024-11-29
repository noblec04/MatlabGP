function ydot = VanDerPolRHS(t,y,Mu)

if nargin<3
    Mu = 100;
end

ydot = [
    y(2);
    Mu*(1-y(1)^2)*y(2)-y(1)
    ];

end