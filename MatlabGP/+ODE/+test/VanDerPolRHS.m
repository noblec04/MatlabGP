function ydot = VanDerPolRHS(t,y,Mu)

if nargin<3
    Mu = 100;
end

y1 = y(1);
y2 = y(2);

ydot = [
    y2;
    Mu*(1-y1^2)*y2-y1
    ];

end