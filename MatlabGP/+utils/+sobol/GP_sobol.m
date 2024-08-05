function [S] = GP_sobol(Z)

thetas = Z.kernel.getHPs();

alphas = Z.alpha;

Kinv = Z.Kinv;

sigmap2 = Z.kernel.scale;

X = (Z.X - Z.lb_x)./(Z.ub_x - Z.lb_x);

for s = 1:size(X,2)
    T1(s) = utils.sobol.Term1(sigmap2,thetas,alphas,Kinv,X,s);
    T2(s) = utils.sobol.Term2(sigmap2,thetas,Kinv,X);
    T3(s) = utils.sobol.Term3(sigmap2,thetas,alphas,X);
    T4(s) = utils.sobol.Term4(sigmap2,Kinv,thetas,alphas,X);
end

S = (T1 - T2 - T3)./(T4 - T2 - T3);

end