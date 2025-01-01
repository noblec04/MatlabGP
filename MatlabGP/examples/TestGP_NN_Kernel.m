clear
clc


A = 0.5; B = 10; C = -5;
ff = @(x) (6*x-2).^2.*sin(12*x-4);

x1 = [0;lhsdesign(300,1);1];
y1 = testFuncs.Forrester(x1,1);

xm = linspace(0,1,20);

%%
layers{1} = NN.FF(1,3);
layers{2} = NN.FF(3,3);
layers{3} = NN.FF(3,3);
layers{4} = NN.FF(3,3);

acts{1} = NN.TANH();
acts{2} = NN.TANH();
acts{3} = NN.TANH();

lss = NN.MAE();

%%

ma = means.zero();

%%

kaNN = NN.NN(layers,acts,lss);

ka = kernels.EQ_NN(kaNN,1,[0.1 0.2 1]);%.periodic(1,10);
ka.signn = 1e-3;

%%

Z = GP(ma,ka);
Z = Z.condition(x1,y1,0,1);

%%
Z = Z.train2();


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