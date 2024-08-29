
clear
close all
clc

lb = [0.5 0.5 2.5*10^(-3)];
ub = [1.5 1.5 7.5*10^(-3)];

xx = lb + (ub - lb).*lhsdesign(50000,3);
yy = testFuncs.StressedPlate(xx,1);

x1 = lb + (ub - lb).*lhsdesign(5,3);
y1 = testFuncs.StressedPlate(x1,1);

x2 = [lb + (ub - lb).*lhsdesign(20,3)];
y2 = testFuncs.StressedPlate(x2,2);

x{1} = x1;
x{2} = x2;

y{1} = y1;
y{2} = y2;

%%
ma = means.linear([1 1 1 1]);
mb = means.linear([1 1 1]);

% ma = means.zero();
% mb = means.zero();

a = kernels.RQ(2,1,[0.1 0.2 0.1 0.3]);%.periodic(1,10);
b = kernels.RQ(2,1,[0.2 0.2 0.2]);
a.signn = eps;
b.signn = 0.01;

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

% layers{1} = NN.FF(3,6);
% layers{2} = NN.FF(6,6);
% layers{3} = NN.FF(6,2);
% 
% acts{1} = NN.SNAKE(1);
% acts{2} = NN.SNAKE(1);
% 
% lss = NN.CE();
% 
% nnc = NN.NN(layers,acts,lss);

%%

% clear delta
% 
% delta(:,1) = double(log(abs(MF.LOO))>median(log(abs(MF.LOO))));
% delta(:,2) = double(log(abs(MF.LOO))<=median(log(abs(MF.LOO))));
% 
% tic
% nnc = nnc.train(x{1},delta);
% toc
% 
% %%
% figure
% utils.isoplotclasses(nnc,1,3)


%%

% mc = means.linear(ones(1,3));%*means.sine(1,10,0,1);
% c = kernels.RQ(2,1,ones(1,3));
% c.signn = 1*10^(-7);
% 
% LOOZ = GP(mc,c);
% LOOZ = LOOZ.condition(x{1},log(abs(MF.LOO)),lb,ub);
% LOOZ = LOOZ.train();


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

C = [50 1];

%%
for jj = 1:100
    
    [xn,Rn] = BO.argmax(@BO.MFSFDelta,MF);
    %[xn,Rn] = BO.argmax(@BO.UCB,LOOZ);
    %[xn,Rn] = BO.argmax(@BO.maxVAR,MF);

    sign(1) = Z{1}.eval_var(xn)/C(1);
    sign(2) = Z{2}.eval_var(xn)/C(2);

    [~,in] = max(sign);

    if in==1
        [x{1},flag] = utils.catunique(x{1},xn);
        if flag
            y{1} = [y{1}; testFuncs.StressedPlate(xn,1)];
        end
    end

    [x{2},flag] = utils.catunique(x{2},xn);
    if flag
        y{2} = [y{2}; testFuncs.StressedPlate(xn,2)];
    end

    for ii = 1:2
        Z{ii} = Z{ii}.condition(x{ii},y{ii},lb,ub);
    end

    MF.GPs = Z;
    MF = MF.condition();

    %LOOZ = LOOZ.condition(x{1},log(abs(MF.LOO)),lb,ub);

    pc(jj,1) = size(x{1},1);
    pc(jj,2) = size(x{2},1);

    R2z(jj) = 1 - mean((yy - Z{1}.eval_mu(xx)).^2)./var(yy);
    RMAEz(jj) = max(abs(yy - Z{1}.eval_mu(xx)))./std(yy);

    R2MF(jj) = 1 - mean((yy - MF.eval_mu(xx)).^2)./var(yy);
    RMAEMF(jj) = max(abs(yy - MF.eval_mu(xx)))./std(yy);

    cost(jj) = C(1)*pc(jj,1)+pc(jj,2);

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

    drawnow

end


%%

xi = lb + (ub - lb).*lhsdesign(size(x{1},1),3);
yi = testFuncs.StressedPlate(xi,1);

Zi = GP(mb,b);
Zi = Zi.condition(xi,yi,lb,ub);
Zi = Zi.train();

%%

1 - mean((yy - Zi.eval_mu(xx)).^2)./var(yy)
max(abs(yy - Zi.eval_mu(xx)))./std(yy)