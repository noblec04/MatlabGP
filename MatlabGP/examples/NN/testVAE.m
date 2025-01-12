
clear
close all
clc

xx = lhsdesign(5,1);
yy = normrnd(forr(xx,0),0*forr(xx,0)+1);

yy = (yy-min(yy(:)))/(max(yy(:))-min(yy(:)));

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

layers1{1} = NN.FF(3,6);
layers1{2} = NN.FF(6,6);
layers1{3} = NN.FF(6,6);

acts1{1} = NN.SWISH(0.8);
acts1{2} = NN.SWISH(1.2);

enc = NN.NN(layers1,acts1,[]);

layers2{1} = NN.FF(3,6);
layers2{2} = NN.FF(6,6);
layers2{3} = NN.FF(6,3);

acts2{1} = NN.SWISH(1.2);
acts2{2} = NN.SWISH(0.8);

dec = NN.NN(layers2,acts2,[]);

lss = NN.VAELoss(1e-6);

AE1 = NN.VAE(enc,dec,lss);

%%

t0 = AE1.getHPs();

[e,de] = AE1.loss(t0,yy,yy);

%%

tic
[AE2,fval,xv,fv] = AE1.train(yy,yy);%,xv,fv
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
plot(yy,yp2,'.')

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

