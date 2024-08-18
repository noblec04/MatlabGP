
clear all
close all
clc

xx = lhsdesign(20,1);
yy = normrnd(forr(xx,0),0*forr(xx,0)+0);

yy = (yy-min(yy(:)))/(max(yy(:))-min(yy(:)));

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

layers1{1} = NN2.FF(3,3);
layers1{2} = NN2.FF(3,2);
layers1{3} = NN2.FF(2,1);
acts1{1} = NN2.SNAKE(1);
acts1{2} = NN2.SNAKE(1);

lss = NN2.MAE();

enc = NN2.NN(layers1,acts1,lss);

layers2{1} = NN2.FF(1,2);
layers2{2} = NN2.FF(2,3);
layers2{3} = NN2.FF(3,3);
acts2{1} = NN2.SNAKE(1);
acts2{2} = NN2.SNAKE(1);

lss = NN2.MAE();

dec = NN2.NN(layers2,acts2,lss);

AE1 = NN2.AE(enc,dec,lss);

%%

t0 = AE1.getHPs();

[e,de] = AE1.loss(t0,yy,yy);

%%

tic
[AE2,fval] = AE1.train(yy,yy);%,xv,fv
toc

%%

yp2 = AE2.forward(yy);


%%
% figure
% plot(fv,'.')
% set(gca,'yscale','log')
% set(gca,'xscale','log')

figure
%plot(xmesh,yp1)
plot(yy,yp2,'.')

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

