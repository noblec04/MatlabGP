
clear 
close all
clc

xx = lhsdesign(50,2);
yy = normrnd(forr(xx),0*forr(xx));

xmesh = lhsdesign(1000,2);
ymesh = forr(xmesh);

nnet = RRNN(NN.SNAKE(8),100,0,1);

%%

tic
[nnet2,Ri] = nnet.train(xx,yy,20);%,xv,fv
toc

%%

yp2 = nnet2.eval(xmesh);


%%
% figure
% plot(fv,'.')
% set(gca,'yscale','log')
% set(gca,'xscale','log')

figure
plot3(xmesh(:,1),xmesh(:,2),ymesh,'.')
hold on
%plot(xmesh,yp1)
plot3(xmesh(:,1),xmesh(:,2),yp2,'.')
plot3(xx(:,1),xx(:,2),yy,'x')

1 - mean((ymesh - yp2).^2)./var(ymesh)

%%

function y = forr(x)


A = 0.5; B = 10; C = -5;

y = (6*x(:,1)-2).^2.*sin(12*x(:,2)-4);

    %{
    y(i,2) = 0.4*(6*x(i)-2).^2.*sin(12*x(i)-4)-x(i)-1;
    y(i,3) = A*(6*x(i)-2).^2.*sin(12*x(i)-4)+B*(x(i)-0.5)-C;
    %}

end

