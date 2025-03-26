
clear
close all
clc

%%
A = 0.5; B = 10; C = -5;
ff = @(x) (6*x-2).^2.*sin(12*x-4);

x1 = [0;lhsdesign(20,1);1];
y1 = ff(x1);

%%
layers{1} = NN.FF(1,3);
layers{2} = NN.FF(3,3);
layers{3} = NN.FF(3,1);

acts{1} = NN.SWISH(1);
acts{2} = NN.SWISH(1);

lss = NN.MAE();

%%
ma = NN.NN(layers,acts,lss);

%%
tic

ma = ma.train(x1,y1,0,1);

toc
%%
ma.lb_x = 0;
ma.ub_x = 1;

a = kernels.EQ(1,1);%.periodic(1,10);
a.signn = 0.001;

%%

Z = GP(ma,a);
Z = Z.condition(x1,y1,0,1);

%%

tic

V = Z.getHPs();

opt = optim.AdamLS(V,'wd',0.00001);
FF = @(x) Z.loss(x); 

for i = 1:200
    
    Vi(:,i) = V;

    [e(i),dV] = Z.loss(V);

    [opt,V] = opt.step(V,FF,dV);

end

Z = Z.setHPs(V);
Z = Z.condition(x1,y1,0,1);

toc