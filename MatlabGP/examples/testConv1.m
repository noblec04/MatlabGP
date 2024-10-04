
clear
close all
clc

xx = lhsdesign(20,1);
yy = normrnd(forr(xx,0),0*forr(xx,0)+1);

yy = (yy-min(yy(:)))/(max(yy(:))-min(yy(:)));

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

%%

layers1{1} = NN.C1(3);
layers1{2} = NN.C1(3);
layers1{3} = NN.FF(3,1);

acts1{1} = NN.SWISH(0.8);
acts1{2} = NN.SWISH(1.2);

lss = NN.MSE();

CNN = NN.NN(layers1,acts1,lss);


%%

t0 = CNN.getHPs();

[e,de] = CNN.loss(t0,yy,yy(:,1));

%%

tic
[CNN2,fval] = CNN.train(yy,yy(:,1));%,xv,fv
toc

%%

yp2 = CNN2.forward(yy);


%%
% figure
% plot(fv,'.')
% set(gca,'yscale','log')
% set(gca,'xscale','log')

figure
%plot(xmesh,yp1)
plot(yy(:,1),yp2,'.')

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

