
clear all
close all
clc

xx = lhsdesign(50,2);
yy = forr(xx,0);

xmesh = lhsdesign(200,2);
ymesh = forr(xmesh,0);

layers{1} = NN.FF(2,50);
layers{2} = NN.FF(50,1);

acts{1} = NN.SWISH(1);

lss = NN.MSE();

nnet = NN.NN(layers,acts,lss);

%%

x0 = nnet.getHPs();

opt = optim.VSGD(x0,'lr',0.01,'lb',0,'ub',1,'gamma',0.1);

xv(:,1) = x0;

%%

for i = 1:500

    [lossvec(i),dy] = nnet.loss(xv(:,i),xx,yy);

    [opt,xv(:,i+1)] = opt.step(xv(:,i),dy);

    figure(1)
    clf(1)
    plot(1:i,lossvec)

    figure(2)
    clf(2)
    pcolor(xv)
    shading flat
end


%%

function y = forr(x,dx)

nx = length(x);

A = 0.5; B = 10; C = -5;

for i = 1:nx
    if x(i)<0.45
        y(i,1) = (6*x(i,1)-2).^2.*sin(12*x(i,2)-4);
    else
        y(i,1) = (6*x(i,1)-2).^2.*sin(12*x(i,2)-4)+dx;
    end

    y(i,2) = 0.4*(6*x(i,1)-2).^2.*sin(12*x(i,2)-4)-x(i,1)-1;
    y(i,3) = A*(6*x(i,2)-2).^2.*sin(12*x(i,1)-4)+B*(x(i,2)-0.5)-C;
end

end

