
clear all
close all
clc

xx = lhsdesign(50,2);
yy = forr(xx,0);

xmesh = lhsdesign(200,2);
ymesh = forr(xmesh,0);

layers{1} = NN.FF(2,5);
layers{2} = NN.FF(5,5);
layers{3} = NN.FF(5,5);
layers{4} = NN.FF(5,1);


acts{1} = NN.SWISH(1);
acts{2} = NN.SWISH(1);
acts{3} = NN.SWISH(1);

lss = NN.MSE();


nnet = NN.NN(layers,acts,lss);

%%
tic
[nnet2,fval,xv,fv] = nnet.train(xx,yy(:,1));
toc

%%

for j = 1:length(xmesh)
    %yp1(:,j) = nnet.predict(xmesh(j,:)');
    yp2(:,j) = nnet2.predict(xmesh(j,:)');
end

%%
figure
plot(fv,'.')
set(gca,'yscale','log')
set(gca,'xscale','log')

figure
for i = 1:1
    subplot(1,3,i)
    plot3(xmesh(:,1),xmesh(:,2),ymesh(:,i),'.')
    hold on
    %plot3(xmesh(:,1),xmesh(:,2),yp1(i,:),'.')
    plot3(xmesh(:,1),xmesh(:,2),yp2(i,:),'.')
    plot3(xx(:,1),xx(:,2),yy(:,i),'x')
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

