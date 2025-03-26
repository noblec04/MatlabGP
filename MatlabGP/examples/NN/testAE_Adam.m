
clear
close all
clc

xx = lhsdesign(200,1);
yy = normrnd(forr(xx,0),0*forr(xx,0) + 0.1);
yt = forr(xx,0);

yy = (yy-min(yy(:)))/(max(yy(:))-min(yy(:)));

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

layers1{1} = NN.FAN(3,4,2);
layers1{2} = NN.FF(4,4);
layers1{3} = NN.FF(4,2);

acts1{1} = NN.SWISH(0.8);
acts1{2} = NN.SWISH(1.2);

lss = NN.MAE();

enc = NN.NN(layers1,acts1,lss);

layers2{1} = NN.FAN(2,4,2);
layers2{2} = NN.FF(4,4);
layers2{3} = NN.FF(4,3);

acts2{1} = NN.SWISH(1.2);
acts2{2} = NN.SWISH(0.8);

lss = NN.MSE();

dec = NN.NN(layers2,acts2,lss);

AE1 = NN.AE(enc,dec,lss);

%%

tic

V = AE1.getHPs();

opt = optim.AdamLS(V(:),'wd',0.001);
FF = @(x) AE1.loss(x,yy,yy); 

for i = 1:800
    
    Vi(:,i) = V;

    [e(i),dV] = AE1.loss(V(:),yy,yy);

    [opt,V] = opt.step(V(:),FF,dV(:));
    
    lr(i) = opt.lr;

    figure(1)
    clf(1)
    subplot(1,2,1)
    plot(e)
    set(gca,'yscale','log')
    set(gca,'xscale','log')

    subplot(1,2,2)
    plot(lr)
    set(gca,'yscale','log')
    set(gca,'xscale','log')

end

AE2 = AE1.setHPs(V);

AE2.lb_x = min(yy);
AE2.ub_x = max(yy);

toc

%%

yp2 = AE2.forward(yy);

zz = AE2.Encoder.forward(yy);


%%
% figure
% plot(fv,'.')
% set(gca,'yscale','log')
% set(gca,'xscale','log')

figure
%plot(xmesh,yp1)
plot(yt,yp2,'.')
hold on
plot(yt,yy,'.')

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

