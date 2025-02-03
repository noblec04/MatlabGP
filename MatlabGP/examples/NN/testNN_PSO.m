
clear
close all
clc

xx = [0;lhsdesign(10,1);1];
yy = normrnd(forr(xx,0),0*forr(xx,0));

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

layers{1} = NN.FAN(1,3,1);
layers{2} = NN.FAN(3,3,1);
layers{3} = NN.FF(3,3);

acts{1} = NN.SWISH(1.2);
acts{2} = NN.SWISH(0.8);

lss = NN.MSE();

nnet = NN.NN(layers,acts,lss);

%%

tic

V = nnet.getHPs();

opt = optim.PSO(10,V);

for i = 1:500
    
    Vi(:,i) = V;

    [opt,V,e(i)] = opt.step(@(x) nnet.loss(x,xx,yy));
    
    figure(1)
    clf(1)
    plot(e)
    set(gca,'yscale','log')
    set(gca,'xscale','log')
    
end

toc

%%
opt = optim.AMSgrad(V,'lr',0.05);%,'wd',0.005);

tic

for i = 1:500
    
    Vi(:,i) = V;

    [e2(i),dV] = nnet.loss(V,xx,yy);
    [opt,V] = opt.step(V,dV);
    
    figure(1)
    clf(1)
    plot(e)
    hold on
    plot(e2)
    set(gca,'yscale','log')
    set(gca,'xscale','log')
    
end

nnet = nnet.setHPs(V);

toc

%%

tic
nnet2 = nnet.train(xx,yy,0,1);
toc


yp2 = nnet.predict(xmesh);

yp3 = nnet2.predict(xmesh);

%%

figure
plot(xmesh,ymesh)
hold on
plot(xmesh,yp2,':')
plot(xmesh,yp3,'--')
plot(xx,yy,'x')

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

