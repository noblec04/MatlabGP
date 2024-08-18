
clear all
close all
clc

xx = lhsdesign(200,2);
yy = normrnd(forr(xx),0*forr(xx).^2+0.5);

xmesh = lhsdesign(1000,2);
ymesh = forr(xmesh);

layers{1} = NN2.FF(2,5);
layers{2} = NN2.FF(5,3);
layers{3} = NN2.FF(3,1);

acts{1} = NN2.SWISH(1);
acts{2} = NN2.SNAKE(1);

lss = NN2.MAE();


nnet = NN2.NN(layers,acts,lss);

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
plot3(xmesh(:,1),xmesh(:,2),ymesh,'.')
hold on
%plot(xmesh,yp1)
plot3(xmesh(:,1),xmesh(:,2),yp2(:,1),'+')

plot3(xx(:,1),xx(:,2),yy,'x','MarkerSize',18)

1 - mean((ymesh - yp2).^2)./var(ymesh)

%%

function y = forr(x)

y = (6*x(:,1)-2).^2.*sin(12*x(:,2)-4);

end

