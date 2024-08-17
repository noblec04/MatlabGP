
A = 0.5; B = 10; C = -5;
ff = @(x) (6*x-2).^2.*sin(12*x-4);

x1 = lhsdesign(20,1);
y1 = ff(x1);

%%
layers{1} = NN2.FF(1,3);
layers{2} = NN2.FF(3,3);
layers{3} = NN2.FF(3,1);

acts{1} = NN2.SWISH(1);
acts{2} = NN2.SWISH(1);

lss = NN2.MAE();

%%
ma = NN2.NN(layers,acts,lss);

%%
ma = ma.train(x1,y1,0,1);

%%
ma.lb_x = 0;
ma.ub_x = 1;

a = kernels.EQ(1,0.1);%.periodic(1,10);
a.signn = 0.01;

%%

Z = GP(ma,a);
Z = Z.condition(x1,y1,0,1);

%%
Z = Z.train();