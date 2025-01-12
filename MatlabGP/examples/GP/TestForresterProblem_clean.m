
clear
close all
clc

D = 1;
nF = 3;

lb = 0;
ub = 1;

xx = lb + (ub - lb).*lhsdesign(500,1);
yy = testFuncs.Forrester(xx,1);

x1 = [0;1];
y1 = testFuncs.Forrester(x1,1);

x2 = [0;lb + (ub - lb).*lhsdesign(1,D);1];%20
y2 = testFuncs.Forrester(x2,2);

x3 = [0;lb + (ub - lb).*lhsdesign(1,D);1];%100
y3 = testFuncs.Forrester(x3,3);

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
b = kernels.EQ(1,ones(1,D));
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
MF = MFGP(Z,ma,a);%
MF = MF.condition(Z);
MF = MF.train();
toc

%%
dec = RL.TS(10,3);

%%
figure
hold on
utils.plotLineOut(Z{1},0,1,'color','r')
utils.plotLineOut(Z{2},0,1,'color','b')
utils.plotLineOut(Z{3},0,1,'color','g')

%%

figure
hold on
utils.plotLineOut(MF,0,1)

%%

1 - mean((yy - Z{1}.eval_mu(xx)).^2)./var(yy)
max(abs(yy - Z{1}.eval_mu(xx)))./std(yy)

1 - mean((yy - MF.eval_mu(xx)).^2)./var(yy)
max(abs(yy - MF.eval_mu(xx)))./std(yy)

%%

C = [500 30 1];%20

for jj = 1:200
   
    [xn,Rn] = BO.argmax(@BO.maxVAR,MF);

    % siggn(1) = abs(MF.expectedReward(xn,1))/(C(1));
    % siggn(2) = abs(MF.expectedReward(xn,2))/(C(2));
    % siggn(3) = abs(MF.expectedReward(xn,3))/(C(3));

    siggn(1) = abs(Z{1}.eval_var(xn))/(C(1));
    siggn(2) = abs(Z{2}.eval_var(xn))/(C(2));
    siggn(3) = abs(Z{3}.eval_var(xn))/(C(3));
     
    [~,nu] = dec.action();

    nu = exp(nu);

    [~,in] = max(0.5*(siggn+nu));

    if in==1
        [x{1},flag] = utils.catunique(x{1},xn,1e-2);
        if flag
            y{1} = [y{1}; testFuncs.Forrester(xn,1)];
        end
    end

    if in==2 || in==1
        [x{2},flag] = utils.catunique(x{2},xn,1e-2);
        if flag
            y{2} = [y{2}; testFuncs.Forrester(xn,2)];
        end
    end

    if in==3 || in==1
        [x{3},flag] = utils.catunique(x{3},xn,1e-2);
        if flag
            y{3} = [y{3}; testFuncs.Forrester(xn,3)];
        end
    end

    for ii = 1:nF
        Z{ii} = Z{ii}.condition(x{ii},y{ii},lb,ub);
    end

    yh1 = MF.eval(xn);

    %MF.GPs = Z;
    MF = MF.condition(Z);

    yh2 = MF.eval(xn);

    Ri(jj) = abs(yh2 - yh1)/C(in);
    Rie(jj) = siggn(in);

    dec = dec.addReward(in,Ri(jj));

    R2z(jj) = 1 - mean((yy - Z{1}.eval_mu(xx)).^2)./var(yy);
    RMAEz(jj) = max(abs(yy - Z{1}.eval_mu(xx)))./std(yy);

    R2MF(jj) = 1 - mean((yy - MF.eval_mu(xx)).^2)./var(yy);
    RMAEMF(jj) = max(abs(yy - MF.eval_mu(xx)))./std(yy);

    [me,ime] = max(abs(yy - MF.eval_mu(xx)));
    maxeMF(jj) = me./abs(yy(ime));

    cost(jj) = C(1)*size(x{1},1)+C(2)*size(x{2},1)+C(3)*size(x{3},1);

    figure(3)
    clf(3)
    utils.plotLineOut(MF,0,1)
    hold on
    plot(xn,yh2,'x')
    % figure(3)
    % clf(3)
    % hold on
    % plot(cost,R2z)
    % plot(cost,R2MF)
    % 
    % figure(4)
    % clf(4)
    % hold on
    % plot(cost,RMAEz)
    % plot(cost,RMAEMF)
    % plot(cost,maxeMF)
    % 
    % figure(5)
    % clf(5)
    % hold on
    % plot(cost,Rie)
    % plot(cost,Ri)
    % 
    % figure(6)
    % clf(6)
    % hold on
    % dec.plotDists

    drawnow

    if RMAEMF(jj)<0.1
        break
    end
end


%%

xi = lb + (ub - lb).*lhsdesign(ceil(cost(end)/C(1)),D);
yi = testFuncs.Forrester(xi,1);

Zi = GP(mb,b);
Zi = Zi.condition(xi,yi,lb,ub);
Zi = Zi.train();

%%

1 - mean((yy - Zi.eval_mu(xx)).^2)./var(yy)
max(abs(yy - Zi.eval_mu(xx)))./std(yy)