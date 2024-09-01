
clear all
close all
clc

D = 3;
nF = 3;

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
ma = means.const(1);
mb = means.linear(ones(1,D));

a = kernels.RQ(2,1,ones(1,D+nF-1));
b = kernels.RQ(2,1,ones(1,D));
a.signn = eps;
b.signn = eps;

%%
tic
for i = 1:nF
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

dec = RL.TS(20,3);


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

C = [100 30 1];%20

for jj = 1:200
   
    [xn,Rn] = BO.argmax(@BO.MFSFDelta,MF);

    siggn(1) = abs(MF.expectedReward(xn,1))/(C(1));
    siggn(2) = abs(MF.expectedReward(xn,2))/(C(2));
    siggn(3) = abs(MF.expectedReward(xn,3))/(C(3));
     
    [~,nu] = dec.action();

    nu = exp(nu);

    [~,in] = max(0.5*(siggn+nu));

    if in==1
        [x{1},flag] = utils.catunique(x{1},xn);
        if flag
            y{1} = [y{1}; testFuncs.Rosenbrock(xn,1)];
        end
    end

    if in==2 || in==1
        [x{2},flag] = utils.catunique(x{2},xn);
        if flag
            y{2} = [y{2}; testFuncs.Rosenbrock(xn,2)];
        end
    end

    if in==3 || in==1
        [x{3},flag] = utils.catunique(x{3},xn);
        if flag
            y{3} = [y{3}; testFuncs.Rosenbrock(xn,3)];
        end
    end

    for ii = 1:nF
        Z{ii} = Z{ii}.condition(x{ii},y{ii},lb,ub);
    end

    yh1 = MF.eval(xn);

    MF.GPs = Z;
    MF = MF.condition();

    yh2 = MF.eval(xn);

    Ri(jj) = abs(yh2 - yh1)/C(in);
    Rie(jj) = siggn(in);

    dec = dec.addReward(in,Ri(jj));

    R2z(jj) = 1 - mean((yy - Z{1}.eval_mu(xx)).^2)./var(yy);
    RMAEz(jj) = max(abs(yy - Z{1}.eval_mu(xx)))./std(yy);

    R2MF(jj) = 1 - mean((yy - MF.eval_mu(xx)).^2)./var(yy);
    RMAEMF(jj) = max(abs(yy - MF.eval_mu(xx)))./std(yy);

    cost(jj) = C(1)*size(x{1},1)+C(2)*size(x{2},1)+C(3)*size(x{3},1);

    figure(3)
    clf(3)
    hold on
    plot(cost,R2z)
    plot(cost,R2MF)

    figure(4)
    clf(4)
    hold on
    plot(cost,RMAEz)
    plot(cost,RMAEMF)

    figure(5)
    clf(5)
    hold on
    plot(cost,Rie)
    plot(cost,Ri)

    drawnow

    if RMAEMF(jj)<0.1
        break
    end
end


%%

xi = lb + (ub - lb).*lhsdesign(size(x{1},1),D);
yi = testFuncs.Rosenbrock(xi,1);

Zi = GP(mb,b);
Zi = Zi.condition(xi,yi,lb,ub);
Zi = Zi.train();

%%

1 - mean((yy - Zi.eval_mu(xx)).^2)./var(yy)
max(abs(yy - Zi.eval_mu(xx)))./std(yy)