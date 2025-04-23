
clear
close all
clc

%%

V1 = [0.3 0.1];

NN = 100;

logp = forr(V1);

kern = kernels.EQ(1,5);

Sampler = utils.ASVGF(@forr,[0 0],[1 1],NN,kern,0.1,-0.1);

tic
for i = 1:1000

    [Sampler,V1,logp] = Sampler.step();

    Vii(:,:,i) = V1;
    logpi(:,i) = logp;

end
toc

%%

%NN/sum(Ni)

%utils.cornerplot(Vii);

%histogram(Vii,50,'Normalization','pdf')

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

