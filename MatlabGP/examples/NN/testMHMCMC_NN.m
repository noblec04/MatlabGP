
clear
close all
clc

%%

xx = [0;lhsdesign(50,1);1];
yy = normrnd(forr(xx,0),0.1*abs(forr(xx,0))+0.4);

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

layers{1} = NN.FAN(1,9,3);
layers{2} = NN.FF(9,6);
layers{3} = NN.FF(6,3);

acts{1} = NN.SWISH(1.2);
acts{2} = NN.SWISH(0.8);

lss = NN.MSE();

nnet = NN.NN(layers,acts,lss);

%%

tic

V = nnet.getHPs();

opt = optim.AdamLS(V,'wd',0);
FF = @(x) nnet.loss(x,xx,yy); 

for i = 1:800
    
    Vi(:,i) = V;

    [e(i),dV] = nnet.loss(V,xx,yy);

    [opt,V] = opt.step(V,FF,dV);

end

nnet = nnet.setHPs(V);

nnet.lb_x = 0;
nnet.ub_x = 1;

toc

figure
subplot(1,2,1)
plot(e)
set(gca,'xscale','log')
set(gca,'yscale','log')
subplot(1,2,2)
plot(Vi')

figure
utils.plotLineOut(nnet,0,1,'CI',false)
hold on
plot(xx,yy,'x')

%%

V1 = nnet.getHPs();

FF = @(x) -1*nnet.loss(x,xx,yy);

NN = 100;

logp = FF(V1);

Sampler = utils.MHMCMC(FF,min(V1)+0*V1,max(V1)+0*V1);

tic
for i = 1:NN

    [V1,logp,N] = Sampler.step(V1,logp);

    Vii(:,i) = V1;
    logpi(i) = logp;
    Ni(i) = N;

end
toc

NN/sum(Ni)

utils.cornerplot(Vii(1:5,:));

%%

[mu,sig] = nnet.MCeval(xmesh,Vii);

%%

figure
hold on
plot(xmesh,mu,'LineWidth',3)
plot(xmesh,mu+2*sig,'--')
plot(xmesh,mu-2*sig,'--')
plot(xx,yy,'x','MarkerSize',18,'LineWidth',3)

%%

function y = forr(x,dx)

nx = length(x);

A = 0.5; B = 10; C = -5;

for i = 1:nx
    if x(i)<0.45
        y(i,1) = (6*x(i)-2).^2.*sin(12*x(i)-4);
    else
        y(i,1) = (6*x(i)-2).^2.*sin(12*x(i)-4)+dx;
    end

    y(i,2) = 0.4*(6*x(i)-2).^2.*sin(12*x(i)-4)-x(i)-1;
    y(i,3) = A*(6*x(i)-2).^2.*sin(12*x(i)-4)+B*(x(i)-0.5)-C;
end

end

