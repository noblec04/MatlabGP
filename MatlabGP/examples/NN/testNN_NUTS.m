
clear
close all
clc

xx = [0;lhsdesign(30,1);1];
yy = normrnd(forr(xx,0),0*forr(xx,0)+0.5);

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

layers{1} = NN.FAN(1,10,4);
layers{2} = NN.FAN(10,10,4);
layers{3} = NN.FF(10,3);

acts{1} = NN.SWISH(1.2);
acts{2} = NN.SWISH(0.8);

lss = NN.MSE();

nnet = NN.NN(layers,acts,lss);

%%

tic

V = nnet.getHPs();

opt = optim.Adam(V,'lr',0.5);

for i = 1:500
    
    Vi(:,i) = V;

    [e(i),dV] = nnet.loss(V,xx,yy);
    [opt,V] = opt.step(V,dV);
    
    % figure(1)
    % clf(1)
    % plot(e)
    % set(gca,'yscale','log')
    % set(gca,'xscale','log')
    
end

nnet = nnet.setHPs(V);

toc

%%

V1 = V;
NUTS1 = utils.NUTS(0.01,3);

tic
for i = 1:100

    [V1, alpha_ave, logp, grad] = NUTS1.step(@(x) nnet.loss(x,xx,yy),V1);

    Vii(:,i) = V1;
    logpi(i) = logp;

end
toc

%%

for i = 1:100
    nnet2 = nnet.setHPs(Vii(:,i));
    yp2(:,:,i) = nnet2.predict(xmesh);
end

figure
hold on
plot(xmesh,squeeze(mean(yp2,3)))
plot(xmesh,squeeze(mean(yp2,3)-2*sqrt(std(yp2,[],3))),':')
plot(xmesh,squeeze(mean(yp2,3)+2*sqrt(std(yp2,[],3))),':')
plot(xx,yy,'x')

figure
hold on
for i = 1:100
    plot(xmesh,squeeze(yp2(:,1,i)))
    plot(xmesh,squeeze(yp2(:,2,i)))
    plot(xmesh,squeeze(yp2(:,3,i)))
end

%%

yp3 = nnet.predict(xmesh);

figure
plot(xmesh,ymesh)
hold on
plot(xmesh,yp3,':')
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

