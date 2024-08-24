
clear all
clc

D = 3;

lb = -2*ones(1,D);
ub = 2*ones(1,D);

xx = lb + (ub - lb).*lhsdesign(50000,D);
yy = testFuncs.Rosenbrock(xx,1);

x1 = lb + (ub - lb).*lhsdesign(5,D);
y1 = testFuncs.Rosenbrock(x1,1);

x2 = [lb + (ub - lb).*lhsdesign(20,D)];
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
ma = means.const([1]);
mb = means.linear(ones(1,D));

a = kernels.RQ(2,1,ones(1,D+2));
b = kernels.RQ(2,1,ones(1,D));
a.signn = eps;
b.signn = eps;

%%
tic
for i = 1:3
    Z{i} = GP(mb,b);
    Z{i} = Z{i}.condition(x{i},y{i},lb,ub);
    Z{i} = Z{i}.train();
end
toc

%%
tic
MF = NLMFGP(Z,ma,a);
MF = MF.condition();
MF = MF.train();
toc

mc = means.linear(ones(1,D));%*means.sine(1,10,0,1);
c = kernels.RQ(2,1,ones(1,D));
c.signn = eps;

LOOZ = GP(mc,c);
LOOZ = LOOZ.condition(x{1},MF.LOO,lb,ub);
LOOZ = LOOZ.train();


%%
figure
hold on
utils.plotSurf(Z{1},1,2,'color','r')
utils.plotSurf(Z{2},1,2,'color','b')
utils.plotSurf(Z{3},1,2,'color','g')

%%

figure
hold on
utils.plotSurf(MF,1,2)

%%

1 - mean((yy - Z{1}.eval_mu(xx)).^2)./var(yy)
max(abs(yy - Z{1}.eval_mu(xx)))./std(yy)

1 - mean((yy - MF.eval_mu(xx)).^2)./var(yy)
max(abs(yy - MF.eval_mu(xx)))./std(yy)

%%
for jj = 1:60
    
    [xn,Rn] = BO.argmax(@BO.UCB,LOOZ);
    %[xn,Rn] = BO.argmax(@BO.MFSFDelta,MF);
    %[xn,Rn] = BO.argmax(@BO.maxVAR,MF);


    x{1} = [x{1}; xn];
    x{2} = [x{2}; xn];
    x{3} = [x{3}; xn];

    y{1} = [y{1}; testFuncs.Rosenbrock(xn,1)];
    y{2} = [y{2}; testFuncs.Rosenbrock(xn,2)];
    y{3} = [y{3}; testFuncs.Rosenbrock(xn,3)];

    for ii = 1:3
        Z{ii} = Z{ii}.condition(x{ii},y{ii},lb,ub);
    end

    MF.GPs = Z;
    MF = MF.condition();

    LOOZ = LOOZ.condition(x{1},MF.LOO,lb,ub);

    R2z(jj) = 1 - mean((yy - Z{1}.eval_mu(xx)).^2)./var(yy);
    RMAEz(jj) = max(abs(yy - Z{1}.eval_mu(xx)))./std(yy);

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


%%

xi = lb + (ub - lb).*lhsdesign(size(x{1},1),D);
yi = testFuncs.Rosenbrock(xn,1);

Zi = GP(mb,b);
Zi = Zi.condition(xi,yi,lb,ub);
Zi = Zi.train();

%%

1 - mean((yy - Zi.eval_mu(xx)).^2)./var(yy)
max(abs(yy - Zi.eval_mu(xx)))./std(yy)