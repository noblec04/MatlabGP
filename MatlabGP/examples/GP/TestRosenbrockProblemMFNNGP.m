
clear all
clc

D = 3;

lb = -2*ones(1,D);
ub = 2*ones(1,D);

xx = lb + (ub - lb).*lhsdesign(50000,D);
yy = testFuncs.Rosenbrock(xx,1);

x1 = lb + (ub - lb).*lhsdesign(5,D);
y1 = testFuncs.Rosenbrock(x1,1);

x2 = [lb + (ub - lb).*lhsdesign(50,D)];
y2 = testFuncs.Rosenbrock(x2,2);

x3 = [lb + (ub - lb).*lhsdesign(100,D)];
y3 = testFuncs.Rosenbrock(x3,3);

x{1} = x1;
x{2} = x2;
x{3} = x3;

y{1} = y1;
y{2} = y2;
y{3} = y3;

%%

layers{1} = NN2.FF(D,6);
layers{2} = NN2.FF(6,3);
layers{3} = NN2.FF(3,1);

acts{1} = NN2.SWISH(1);
acts{2} = NN2.SWISH(1);

lss = NN2.MAE();

nnet = NN2.NN(layers,acts,lss);

%%

for i = 1:3
    NN{i} = nnet.train(x{i},y{i},lb,ub);
end

%%
ma = means.const(1);
a = kernels.RQ(2,1,ones(1,D+2));
a.signn = eps;


%%

tic
MF = NLMFGP(NN,ma,a);
MF = MF.condition();
MF = MF.train();
toc

%%

1 - mean((yy - NN{1}.eval_mu(xx)).^2)./var(yy)
max(abs(yy - NN{1}.eval_mu(xx)))./std(yy)

1 - mean((yy - MF.eval_mu(xx)).^2)./var(yy)
max(abs(yy - MF.eval_mu(xx)))./std(yy)

%%
figure
hold on
utils.plotSurf(NN{1},1,2,'color','r','CI',false)
utils.plotSurf(NN{2},1,2,'color','b','CI',false)
utils.plotSurf(NN{3},1,2,'color','g','CI',false)

%%

figure
hold on
utils.plotSurf(MF,1,2,'CI',false)

%%
for jj = 1:60
    
    [xn,Rn] = BO.argmax(@BO.MFSFDelta,MF);
    %[xn,Rn] = BO.argmax(@BO.maxVAR,MF);


    x{1} = [x{1}; xn];
    x{2} = [x{2}; xn];
    x{3} = [x{3}; xn];

    y{1} = [y{1}; testFuncs.Rosenbrock(xn,1)];
    y{2} = [y{2}; testFuncs.Rosenbrock(xn,2)];
    y{3} = [y{3}; testFuncs.Rosenbrock(xn,3)];

    for i = 1:3
        NN{i} = NN{i}.train(x{i},y{i},lb,ub);
    end

    MF.GPs = NN;
    MF = MF.condition();
    MF = MF.train();

    R2z(jj) = 1 - mean((yy - MF.GPs{1}.eval_mu(xx)).^2)./var(yy);
    RMAEz(jj) = max(abs(yy - MF.GPs{1}.eval_mu(xx)))./std(yy);

    R2MF(jj) = 1 - mean((yy - MF.eval_mu(xx)).^2)./var(yy);
    RMAEMF(jj) = max(abs(yy - MF.eval_mu(xx)))./std(yy);

    figure(3)
    clf(3)
    hold on
    plot(1:jj,R2z)
    plot(1:jj,R2MF)

    figure(4)
    clf(4)
    hold on
    plot(1:jj,RMAEz)
    plot(1:jj,RMAEMF)

    drawnow

    if RMAEMF(jj)<0.05
        break
    end

end