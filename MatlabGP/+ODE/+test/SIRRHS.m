function ydot = SIRRHS(t,y,beta,gamma)

if nargin<4
    beta = 0.4;
    gamma = 0.04;
end

S = y(1);
I = y(2);
R = y(3);
N = S+I+R;

ydot = [
    -beta*(I*S)/N;
    beta*(I*S)/N - gamma*I;
    gamma*I
    ];
end