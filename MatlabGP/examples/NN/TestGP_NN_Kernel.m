clear
clc


A = 0.5; B = 10; C = -5;
ff = @(x) (6*x-2).^2.*sin(12*x-4);

x1 = [0;lhsdesign(100,1);1];
y1 = testFuncs.Forrester(x1,1);

xm = linspace(0,1,20);

%%
layers{1} = NN.FF(1,3);
layers{2} = NN.FF(3,3);
layers{2} = NN.FF(3,2);

acts{1} = NN.SWISH(1.2);
acts{2} = NN.SWISH(0.8);

lss = NN.MAE();

%%

ma = means.zero();

%%

kaNN = NN.NN(layers,acts,lss);

kaNN.lb_x = 0;
kaNN.ub_x = 1;

ka = kernels.EQ_NN(kaNN,1,[1 1]);%.periodic(1,10);
ka.signn = 1e-3;

%%

Z = GP(ma,ka);
Z = Z.condition(x1,y1,0,1);

ky = Z.kernel.NN.eval(xm');
figure
plot(xm,ky)

%%
Z = Z.train2();

%%
figure
utils.plotLineOut(Z,0,1)

ky = Z.kernel.NN.eval(xm');
figure
plot(xm,ky)


%%
%{

%%

layers{1} = NN.FF(1,3);
layers{2} = NN.FF(3,5);
layers{3} = NN.FF(5,5);
layers{4} = NN.FF(5,1);

acts{1} = NN.SWISH(0.8);
acts{2} = NN.SWISH(0.8);
acts{3} = NN.SWISH(1.2);

lss = NN.MAE();

nnet = NN.NN(layers,acts,lss);

%%

nnet = nnet.train(x1,y1,0,1);
%}