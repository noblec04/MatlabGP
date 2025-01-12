
clear
clc

xx = [0;lhsdesign(5,1);1];
yy = normrnd(forr(xx,0),0*forr(xx,0)+0);

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

layers{1} = NN.FF(1,3);
layers{2} = NN.FAN(3,6,1);
layers{3} = NN.FF(6,3);

acts{1} = NN.SNAKE(2);
acts{2} = NN.SNAKE(1);

lss = NN.MAE();

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
plot(xmesh,ymesh,'.')
hold on
%plot(xmesh,yp1)
plot(xmesh,yp2)
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

