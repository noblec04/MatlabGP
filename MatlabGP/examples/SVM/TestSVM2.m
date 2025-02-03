clear
clc

lb = [-2 -2];
ub = [2 2];

xmesh = lb + (ub - lb).*lhsdesign(1000,2);
ymesh = Rosenbrock_disc(xmesh);

xx = lb + (ub - lb).*lhsdesign(20,2);
yy = Rosenbrock_disc(xx);

layers{1} = NN.FF(2,6);
layers{2} = NN.FAN(6,6,2);
layers{3} = NN.FF(6,1);

acts{1} = NN.SWISH(0.8);
acts{2} = NN.SWISH(1.2);

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

plot3(xmesh(:,1),xmesh(:,2),ymesh,'.')
hold on
plot3(xmesh(:,1),xmesh(:,2),yp2,'.')
plot3(xx(:,1),xx(:,2),yy,'.')


%%

function y = Rosenbrock_disc(x)

d = size(x,2);
sum = 0;
for ii = 1:(d-1)
    xi = x(:,ii);
    xnext = x(:,ii+1);
    new = 100*(xnext-xi.^2).^2 + (xi-1).^2;
    sum = sum + new;
end

y = sum;%/7210;

for i = 1:size(x,1)
    if norm(x(i,:))>1
        y(i) = y(i) + 1000;
    end
end

y = y/7210;
end