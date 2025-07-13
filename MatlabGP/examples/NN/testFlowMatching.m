
clear
close all
clc

%%

layers{1} = NN.FF(2,6);
layers{2} = NN.FF(6,6);
layers{3} = NN.FF(6,1);

acts{1} = NN.SWISH(1.2);
acts{2} = NN.SWISH(0.8);

lss = NN.MMD(1);

nnet = NN.NN(layers,acts,lss);

%%

nt = 5;
nx = 100;

yy = normrnd(6,0.1,nx,1);
tt = linspace(0,1,nt);

%%
V = nnet.getHPs();

xx = randn(nx,1);

for i = 1:nt
    xx = xx + nnet.set_eval(V,[xx 0*xx(:,1)+i]);
end

ee = lss.forward(xx,yy);
de = full(getderivs(ee));

%%

opt = optim.diffGrad(V,'lr',0.001);
FF = @(x) nnet.loss(x,xx,yy); 

for i = 1:600
    
    Vi(:,i) = V;

    xx = randn(nx,1);

    for j = 1:nt
        xx = xx + nnet.set_eval(V,[xx 0*xx(:,1)+tt(j)]);
    end

    ee = lss.forward(xx,yy);
    dV = full(getderivs(ee));
    e(i) = getvalue(ee);

    [opt,V] = opt.step(V,dV);

    figure(1)
    clf(1)
    plot(e,'+')
    set(gca,'yscale','log')
    set(gca,'xscale','log')

end