
clear all
clc

lb = [0.5 0.5 2.5*10^(-3)];
ub = [1.5 1.5 7.5*10^(-3)];

xx = lb + (ub - lb).*lhsdesign(50000,3);
yy = StressedPlate(xx,1);

x1 = lb + (ub - lb).*lhsdesign(5,3);
y1 = StressedPlate(x1,1);

x2 = [lb + (ub - lb).*lhsdesign(20,3)];
y2 = StressedPlate(x2,2);

x{1} = x1;
x{2} = x2;

y{1} = y1;
y{2} = y2;

%%
ma = means.linear([1 1 1 1]);
mb = means.linear([1 1 1]);

a = kernels.EQ(1,[0.1 0.2 0.1 0.3]);%.periodic(1,10);
b = kernels.EQ(1,[0.2 0.2 0.2]);
a.signn = 0;
b.signn = 0;

%%
tic
for i = 1:2
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


%%
figure
hold on
utils.plotLineOut(Z{1},(lb+ub)/2,1,'color','r')
utils.plotLineOut(Z{2},(lb+ub)/2,1,'color','b')

%%

figure
hold on
utils.plotLineOut(MF,(lb+ub)/2,1)

%%

1 - mean((yy - Z{1}.eval_mu(xx)).^2)./var(yy)
max(abs(yy - Z{1}.eval_mu(xx)))./std(yy)

1 - mean((yy - MF.eval_mu(xx)).^2)./var(yy)
max(abs(yy - MF.eval_mu(xx)))./std(yy)

%%
for jj = 1:60
    
    [xn,Rn] = BO.argmax(@BO.maxVAR,MF);

    x{1} = [x{1}; xn];
    x{2} = [x{2}; xn];

    y{1} = [y{1}; StressedPlate(xn,1)];
    y{2} = [y{2}; StressedPlate(xn,2)];

    for ii = 1:2
        Z{ii} = Z{ii}.condition(x{ii},y{ii},lb,ub);
    end

    MF.GPs = Z;
    MF = MF.condition();

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

end


%%

xi = lb + (ub - lb).*lhsdesign(size(x{1},1),3);
yi = StressedPlate(xi,1);

Zi = GP(mb,b);
Zi = Zi.condition(xi,yi,lb,ub);
Zi = Zi.train();

%%

1 - mean((yy - Zi.eval_mu(xx)).^2)./var(yy)
max(abs(yy - Zi.eval_mu(xx)))./std(yy)