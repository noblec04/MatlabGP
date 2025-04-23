
clear
close all
clc

xx = lhsdesign(200,1);
yy = normrnd(forr(xx,0),0*forr(xx,0) + 0.1);
yt = forr(xx,0);

lby = min(yy(:));
uby = max(yy(:));

yy = (yy-min(yy(:)))/(max(yy(:))-min(yy(:)));

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

layers1{1} = NN.FF(3,6);
layers1{2} = NN.FF(6,6);
layers1{3} = NN.FF(6,2);

acts1{1} = NN.SWISH(0.8);
acts1{2} = NN.SWISH(1.2);

lss = NN.MAE();

enc = NN.NN(layers1,acts1,lss);

layers2{1} = NN.FF(1,6);
layers2{2} = NN.FF(6,6);
layers2{3} = NN.FF(6,3);

acts2{1} = NN.SWISH(1.2);
acts2{2} = NN.SWISH(0.8);

lss = NN.VAELoss(0.5);

dec = NN.NN(layers2,acts2,lss);

AE1 = NN.VAE(enc,dec,lss);

%%

tic

V = AE1.getHPs();

opt = optim.PSO(100,V(:));
FF = @(x) AE1.loss(x,yy,yy); 

for i = 1:800

    Vi(:,i) = V;

    [opt,V,e(i)] = opt.step(FF);

    figure(1)
    clf(1)
    plot(e)
    set(gca,'yscale','log')
    set(gca,'xscale','log')

end

AE2 = AE1.setHPs(V);

AE2.lb_x = min(yy);
AE2.ub_x = max(yy);

toc

%%

yp2 = AE2.forward(yt);

zz = AE2.Encoder.forward(yt);


%%
% figure
% plot(fv,'.')
% set(gca,'yscale','log')
% set(gca,'xscale','log')

figure
%plot(xmesh,yp1)
plot(yt,lby + (uby - lby).*yp2,'.')
hold on
plot(yt,lby + (uby - lby).*yy,'.')

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

