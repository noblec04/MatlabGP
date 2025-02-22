function [alpha] = HVUCB(Z,x)

lb_y = min(Z.Y);
ub_y = max(Z.Y);

Y = (Z.Y - lb_y)./(ub_y - lb_y);

A = utils.ParetoFront(Y);
%HV = utils.hypervolume(Y(A==1,:),max(Y));
HV = sum(Y(A==1,:),'all');

yn = Z.UCB(x);

Yn = [Z.Y;yn];

Yn = (Yn - lb_y)./(ub_y - lb_y);

An = utils.ParetoFront(Yn);
%HVn = utils.hypervolume(Yn(An==1,:),max(Y));
HVn = sum(Yn(An==1,:),'all');

alpha = HVn - HV;

alpha = -1*alpha;

end