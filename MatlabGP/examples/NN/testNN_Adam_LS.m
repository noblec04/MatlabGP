
clear
close all
clc

xx = [0;lhsdesign(10,1);1];
yy = normrnd(forr(xx,0),0*abs(forr(xx,0)));

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

layers{1} = NN.FAN(1,6,2);
layers{2} = NN.FF(6,6);
layers{3} = NN.FF(6,3);

acts{1} = NN.SWISH(1.2);
acts{2} = NN.SWISH(0.8);

lss = NN.MAE();

nnet = NN.NN(layers,acts,lss);

%%

tic

V = nnet.getHPs();

opt = optim.AdamLS(V,'wd',0.001);
FF = @(x) nnet.loss(x,xx,yy); 

for i = 1:800
    
    Vi(:,i) = V;

    [e(i),dV] = nnet.loss(V,xx,yy);

    [opt,V] = opt.step(V,FF,dV);
    
    % lr(i) = opt.lr;
    % 
    % figure(1)
    % clf(1)
    % subplot(1,2,1)
    % plot(e)
    % set(gca,'yscale','log')
    % set(gca,'xscale','log')
    % 
    % subplot(1,2,2)
    % plot(lr)
    % set(gca,'yscale','log')
    % set(gca,'xscale','log')
end

nnet = nnet.setHPs(V);

nnet.lb_x = 0;
nnet.ub_x = 1;

toc

%%

nnet2 = NN.NN(layers,acts,lss);

tic
nnet2 = nnet2.train(xx,yy,0,1);
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

