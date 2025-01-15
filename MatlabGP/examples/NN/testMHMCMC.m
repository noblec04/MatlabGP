
clear
close all
clc

%%

V1 = [0.3 0.1];

NN = 10000;

logp = forr(V1);

Sampler = utils.MHMCMC(@forr,[0 0],[1 1]);

tic
for i = 1:NN

    [V1,logp,N] = Sampler.step(V1,logp);

    Vii(:,i) = V1;
    logpi(i) = logp;
    Ni(i) = N;

end
toc

NN/sum(Ni)

utils.cornerplot(Vii);

%histogram(Vii,50,'Normalization','pdf')

%%
[y,dy] = forr([0 0.1])

%%

function [y,dy] = forr(x)

if nargout==2
    x = AutoDiff(x);
end

y = (((6*x(:,1)-2).^2.*sin(12*x(:,2)-4)).^2)/300;

if nargout==2
    dy = full(getderivs(y));
    y = getvalue(y);
end

end

