
clear all
close all
clc

xx = [0;lhsdesign(10,1);1];
yy = forr(xx,0);

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

layers{1} = NN.FF(1,6);
layers{2} = NN.FF(6,10);
layers{3} = NN.FF(10,3);

acts{1} = NN.SNAKE(1);
acts{2} = NN.SNAKE(1);

lss = NN.MSE();


nnet = NN.NN(layers,acts,lss);

%%
for ii = 1:20
    tic
    [nnet2{ii},fval(ii),xv,fv] = nnet.train(xx,yy);
    toc
end

%%
for ii = 1:20
    for j = 1:length(xmesh)
        %yp1(:,j) = nnet.predict(xmesh(j,:)');
        yp2(:,j,ii) = nnet2{ii}.predict(xmesh(j));
    end
end

%%
figure
plot(fv,'.')
set(gca,'yscale','log')
set(gca,'xscale','log')

figure
plot(xmesh,ymesh)
hold on
%plot(xmesh,yp1)
plot(xmesh,squeeze(mean(yp2,3)))
plot(xmesh,squeeze(mean(yp2,3))+2*squeeze(std(yp2,[],3)),'--')
plot(xmesh,squeeze(mean(yp2,3))-2*squeeze(std(yp2,[],3)),'--')
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

