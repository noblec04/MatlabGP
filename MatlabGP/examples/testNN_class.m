
clear
close all
clc

xx = lhsdesign(20,2);
yy = normrnd(forr(xx,0),0*forr(xx,0));

yc(:,1) = double(yy>0&yy<=1);
yc(:,2) = double(yy<=0);
yc(:,3) = double(yy>1);


xmesh = lhsdesign(1000,2);
ymesh = forr(xmesh,0);

layers{1} = NN.FF(2,3);
layers{2} = NN.FF(3,6);
layers{3} = NN.FF(6,3);

acts{1} = NN.SNAKE(1);
acts{2} = NN.SNAKE(1);

lss = NN.CE();

nnet = NN.NN(layers,acts,lss);

%%

tic
[nnet2,fval] = nnet.train(xx,yc);%,xv,fv
toc

%%

yp2 = nnet2.predict(xmesh);

yp3 = utils.softargmax(yp2);


%%
% figure
% plot(fv,'.')
% set(gca,'yscale','log')
% set(gca,'xscale','log')

figure
plot3(xmesh(:,1),xmesh(:,2),yp3,'.')
hold on
plot3(xx(:,1),xx(:,2),yc,'+')

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
end

end

