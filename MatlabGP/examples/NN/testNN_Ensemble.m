
clear all
close all
clc

xx = [0;lhsdesign(8,1);1];
yy = forr(xx,0);

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

layers{1} = NN.FF(1,8);
layers{2} = NN.FF(8,8);
layers{3} = NN.FF(8,3);

acts{1} = NN.TANH();
acts{2} = NN.SWISH(1);

lss = NN.MSE();

nE = 20;
for i = 1:nE
    nnet{i} = NN.NN(layers,acts,lss);
end

%%
tic
for i = 1:nE
    [nnet2{i}] = nnet{i}.train(xx,yy);
end
toc

%%
for i = 1:nE
    yp2(:,:,i) = nnet2{i}.predict(xmesh);
end

%%

figure
plot(xmesh,ymesh)
hold on
%plot(xmesh,yp1)
plot(xmesh,squeeze(mean(yp2,3)))
plot(xmesh,squeeze(mean(yp2,3))+squeeze(2*std(yp2,0,3)),'k--')
plot(xmesh,squeeze(mean(yp2,3))-squeeze(2*std(yp2,0,3)),'k--')
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

