
clear
close all
clc

xx = [0;lhsdesign(10,1);1];
yy = normrnd(forr(xx,0),0*forr(xx,0));

xmesh = linspace(0,1,100)';
ymesh = forr(xmesh,0);

layers{1} = NN.FF(1,3);
layers{2} = NN.FF(3,3);
layers{3} = NN.FF(3,3);

acts{1} = NN.SWISH(1.8);
acts{2} = NN.SWISH(0.8);

lss = NN.MAE();

nnet = NN.NN(layers,acts,lss);

%%

V = nnet.getHPs();

opt = optim.Adam(V,'lr',0.01);

%opt = optim.VSGD(V,'lr',0.5,'gamma',1*10^(-3));

N = 3;

eMv = 0;
iim = 0;

n = 0;
figure(1)

for i = 1:1000
    xt = xx;
    yt = yy;
    M=0;
    dV = 0*V;
    eM = 0;
   
    while size(xt,1)>0
        n=n+1;
        M=M+1;
        itrain = randsample(size(xt,1),min(N,size(xt,1)));
       
        xtt = xt(itrain,:);
        ytt = yt(itrain,:);
       
        xt(itrain,:)=[];
        yt(itrain,:)=[];

        [e(n),dVi] = nnet.loss(V,xtt,ytt);
       
        %e(n) = e(n)/length(itrain);
        eM = eM + e(n);

        %[opt,V] = opt.step(V,dVi);
       
        dV = dV + dVi;
       
        utils.sfigure(1)
        clf(1)
        hold on
        plot(e,'LineWidth',3)
        plot(iim,eMv,'x','MarkerSize',15,'LineWidth',3)
        set(gca,'yscale','log')
        set(gca,'xscale','log')
        drawnow
       
    end
   
    eMv(i) = eM/M;
    iim(i) = n;

    [opt,V] = opt.step(V,dV/M);
   
end

nnet = nnet.setHPs(V);

%%

yp2 = nnet.predict(xmesh);


%%

figure
plot(xmesh,ymesh)
hold on
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

