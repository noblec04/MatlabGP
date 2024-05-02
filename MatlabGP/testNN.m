
clear all
close all
clc

xx = lhsdesign(10,1);
yy = forr(xx,0);

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

layers{1} = NN.FF(1,10);
layers{2} = NN.FF(10,10);
layers{3} = NN.FF(10,1);

acts{1} = NN.SWISH(1);
acts{2} = NN.SWISH(1);

lss = NN.MSE();


nnet = NN.NN(layers,acts,lss);

%%
tic
[nnet2,fval,xv,fv] = nnet.train(xx,yy);
toc

%%

for j = 1:length(xmesh)
    yp1(j) = nnet.forward(xmesh(j,:));
    yp2(j) = nnet2.forward(xmesh(j,:));
end

%%
figure
plot(fv,'.')
set(gca,'yscale','log')
set(gca,'xscale','log')

figure
plot(xmesh,ymesh)
hold on
plot(xmesh,yp1)
plot(xmesh,yp2)
plot(xx,yy,'x')

%%

function y = forr(x,dx)

nx = length(x);

for i = 1:nx
    if x(i)<0.45
        y(i,1) = (6*x(i)-2).^2.*sin(12*x(i)-4);
    else
        y(i,1) = (6*x(i)-2).^2.*sin(12*x(i)-4)+dx;
    end
end

end

