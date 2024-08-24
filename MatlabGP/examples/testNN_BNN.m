
clear all
close all
clc

xx = [0;lhsdesign(100,1);1];
yy = normrnd(forr(xx,0),0.01*forr(xx,0).^2+0.5);

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

layers{1} = NN.FF(1,3);
layers{2} = NN.FF(3,3);
layers{3} = NN.FF(3,2);

acts{1} = NN.SWISH(1);
acts{2} = NN.SNAKE(1);

lss = NN.NLL();

nnet = NN.NN(layers,acts,lss);

%%

tic
[nnet2,fval] = nnet.train(xx,yy);%,xv,fv
toc

%%
yp2 = nnet2.predict(xmesh);


%%
% figure
% plot(fv,'.')
% set(gca,'yscale','log')
% set(gca,'xscale','log')

figure
plot(xmesh,ymesh)
hold on
%plot(xmesh,yp1)
plot(xmesh,yp2(:,1))
plot(xmesh,yp2(:,1) + 2*sqrt(exp(yp2(:,2))))
plot(xmesh,yp2(:,1) - 2*sqrt(exp(yp2(:,2))))

plot(xx,yy,'x')

%%

function y = forr(x,dx)

nx = length(x);

A = 0.5; B = 10; C = -5;

for i = 1:nx
    if x(i)<0.45
        y(i) = (6*x(i)-2).^2.*sin(12*x(i)-4);
    else
        y(i) = (6*x(i)-2).^2.*sin(12*x(i)-4)+dx;
    end

end

end

