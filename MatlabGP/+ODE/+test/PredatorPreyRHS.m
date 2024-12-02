function ydot = PredatorPreyRHS(t,y,alpha,beta,gamma,delta)

if nargin<6
    alpha = 0.1;
    beta = 0.4;
    gamma = 1.1;
    delta = 0.4;
end

x_prey = y(1);
x_pred = y(2);

ydot = [
    alpha*x_prey - beta*x_prey*x_pred;
    -gamma*x_pred + delta*x_prey*x_pred;
    ];
end