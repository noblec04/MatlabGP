function [E,V] = BayesQuadNormal(scale,thetas,K11,X,res,mus,sigmas)

Thetas = diag(thetas);
Sigmas = diag(sigmas);

II = eye(numel(sigmas));

Am = det(Thetas\Sigmas + II)^(-1/2);

Mu1 = X - mus;

Bm = exp(-0.5*Mu1.*((Thetas+Sigmas)\Mu1));

zm = scale*Am*Bm;

Av = det(2*Thetas\Sigmas + II)^(-1/2);

E = zm'*(K11\res);

V = scale*Av - zm'*(K11\zm);

end