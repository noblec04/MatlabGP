
clear
close all
clc

lb = [-5 -5];
ub = [5 5];

%[X,Y] = ndgrid(linspace(-5,5,16),linspace(-5,5,16));

%XX = [X(:) Y(:)];

XX = lb + (ub - lb).*lhsdesign(300,2);

xx = XX;
yy = testFuncs.SmoothCircle(xx,5);

ma = means.const(1);
ka = kernels.Matern12(1,1);

z=GP(ma,ka);

z = z.condition(xx,yy,lb, ub);

z = z.train();

figure
utils.contourf(z,[0 0],2,1,'nL',10);

%%

layers{1} = NN.FF(200,64);
layers{2} = NN.FF(64,64);
layers{3} = NN.FF(64,100);

acts{1} = NN.SNAKE(2);
acts{2} = NN.SNAKE(1);

lss = NN.MAE();

nnet = NN.NN(layers,acts,lss);

[X,Y] = ndgrid(linspace(-5,5,10),linspace(-5,5,10));

XX = [X(:) Y(:)];

[mu,sig] = z.eval(XX);

out = reshape(utils.softargmax(nnet.forward([mu(:);sig(:)]'))',size(X));

figure
pcolor(out)
shading flat
colorbar