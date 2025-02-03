
clear
close all
clc

D = 1;
nF = 3;

lb = 0;
ub = 1;

xx = lb + (ub - lb).*lhsdesign(500,1);
yy = testFuncs.Forrester(xx,1);

x1 = [0;lhsdesign(1,1);1];
y1 = testFuncs.Forrester(x1,1);

x2 = [x1;lb + (ub - lb).*lhsdesign(5,1)];%20
y2 = testFuncs.Forrester(x2,2);

x3 = [x2;lb + (ub - lb).*lhsdesign(8,1)];%100
y3 = testFuncs.Forrester(x3,3);


x{1} = x1;
x{2} = x2;
x{3} = x3;

y{1} = y1;
y{2} = y2;
y{3} = y3;

%%

ma = means.const(1);

ka = kernels.EQ(1,ones(1,D+nF-1));
ka.signn = eps;

%%
tic
for i = 1:nF
    Z{i} = GP(ma,ka);
    Z{i} = Z{i}.condition(x{i},y{i},lb,ub);
    Z{i} = Z{i}.train();
end
toc

%%
tic
MF = MFGP(Z,ma,ka);%
MF = MF.condition(Z);
MF = MF.train();
toc

%%

figure
utils.plotLineOut(Z{1},0,1)

figure
utils.plotLineOut(MF,0,1)

%%
[xn,Rn] = BO.argmax(@BO.maxSC,MF)
yn{1} = testFuncs.Forrester(xn,1);
yn{2} = testFuncs.Forrester(xn,2);
yn{3} = testFuncs.Forrester(xn,3);


%%

tic
MFn = MF.resolve(xn,yn,1);
toc

figure
utils.plotLineOut(MFn,0,1)

fprintf('SF')
[E,V] = Z{1}.BayesQuad()
fprintf('MF')
[E,V] = MF.BayesQuad()
fprintf('MFn')
[E,V] = MFn.BayesQuad()
fprintf('True')
mean(yy)