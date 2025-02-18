
clear
close all
clc

xmesh = linspace(0,1,30);

figure
plot(xmesh,forr(xmesh))

%%

V1 = 0.9;
NUTS1 = utils.NUTS(0.001,4);

tic
for i = 1:1000

    [V1, alpha_ave, logp, grad] = NUTS1.step(@forr,V1);

    Vii(:,i) = V1;
    logpi(i) = logp;

end
toc

%%

tic

M=1; %number of model parameters
Nwalkers=40; %number of walkers/chains.
minit=randn(M,Nwalkers);

[models,logP]=utils.gwmcmc(minit,@forr,1000,'StepSize',10,'burnin',.1);

toc


%%

function [y,dy] = forr(x)

if nargout==2
    x = AutoDiff(x);
end

y = (((6*x-2).^2.*sin(12*x-4)).^2)/300;

y(x>1)=0;
y(x<0)=0;

if nargout==2
    dy = getderivs(y);
    y = getvalue(y);
end

end

