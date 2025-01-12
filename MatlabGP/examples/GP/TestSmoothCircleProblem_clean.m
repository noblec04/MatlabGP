
clear
close all
clc

lb = [-5 -5];
ub = [5 5];

x0 = [1 2];
R = 5;

xx = lb + (ub - lb).*lhsdesign(1000,2);
yy = testFuncs.SmoothCircle(xx,x0,R);

x1 = lb + (ub - lb).*lhsdesign(10,2);
y1 = testFuncs.SmoothCircle(x1,x0,R);


%%
ma = means.const(1);

a = kernels.Matern12(1,0.5);
a.signn = eps;

%%
tic
Z = GP(ma,a);
Z = Z.condition(x1,y1,lb,ub);
Z = Z.train();
toc

%%
figure
hold on
utils.contourf(Z,(lb+ub)/2,1,2,'color','r')

%%

1 - mean((yy - Z.eval_mu(xx)).^2)./var(yy)
max(abs(yy - Z.eval_mu(xx)))./std(yy)

%%

xn = [];

for jj = 1:200
    
    % for k = 1:10
    %     [xni(k,:),Rni(k)] = BO.argmax(@BO.UCB,Z);
    % end

    %[~,in] = max(Rni);
    %xn = xni(in,:);

    xn = BO.TrustRegion2(Z);

    % figure(6)
    % clf(6)
    % histogram(Rni)

    [x1,flag] = utils.catunique(x1,xn,1e-6);
    if flag
        y1 = [y1; testFuncs.SmoothCircle(xn,x0,R)];
    end

    yh1 = Z.eval(xn);

    Z = Z.condition(x1,y1,lb,ub);
    if jj<10
        Z = Z.train();
    end

    yh2 = Z.eval(xn);

    Ri(jj) = abs(yh2 - yh1);

    R2z(jj) = 1 - mean((yy - Z.eval_mu(xx)).^2)./var(yy);
    RMAEz(jj) = max(abs(yy - Z.eval_mu(xx)))./std(yy);

    figure(1)
    clf(1)
    utils.contourf(Z,(lb+ub)/2,1,2,'color','r')

    figure(3)
    clf(3)
    hold on
    plot(R2z)

    figure(4)
    clf(4)
    hold on
    plot(RMAEz)

    figure(5)
    clf(5)
    hold on
    plot(Ri)

    drawnow

    if RMAEz(jj)<0.1
        break
    end

end