function [x,R] = argmaxcon(FF,Z,C,x0)

lb = Z.lb_x;
ub = Z.ub_x;

if nargin<3||isempty(x0)
    x0 = lb + (ub-lb).*rand(1,length(lb));
end

try
    opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,'Display','off');

    [x,R] = fmincon(@(x) FF(Z,x),x0,[],[],[],[],lb,ub,@(x) nlcon(C,x),opts);

catch

    opts = optimoptions('fmincon','SpecifyObjectiveGradient',false,'Display','off');

    [x,R] = fmincon(@(x) FF(Z,x),x0,[],[],[],[],lb,ub,@(x) nlcon(C,x),opts);

end

R = -1*R;

end

function [C,Ceq,GC,GCeq] = nlcon(C,x)

C = C.eval(x);
Ceq = [];
GC = C.eval_grad(x);
GCeq = [];

end