
function [ins, outs, indices, dRMSEz, RMSEz, RMAEz] = TestSmoothCircleProblem_NN(Decider,plot)

lb = [-5 -5];
ub = [5 5];

R = 5*(rand-0.5);

x0 = 2*(rand-0.5);
y0 = 3*(rand-0.5);

X0 = [x0 y0];

ff = @(x) testFuncs.SmoothCircle(x,X0,R);

xx = lb + (ub - lb).*lhsdesign(10000,2);
yy = ff(xx);

x1 = lb + (ub - lb).*lhsdesign(10,2);
y1 = ff(x1);


%%
ma = means.const(1);

a = kernels.Matern12(1,0.5);
a.signn = eps;

%%

Z = GP(ma,a);
Z = Z.condition(x1,y1,lb,ub);
Z = Z.train();

%%
if plot
    figure
    hold on
    utils.contourf(Z,(lb+ub)/2,1,2,'color','r')
end

%%

RMSEz0 = mean((yy - Z.eval_mu(xx)).^2)./var(yy);

%%

xn = [];

[X,Y] = ndgrid(linspace(-5,5,10),linspace(-5,5,10));

 X = X(:);
 Y = Y(:);
    
 XX = [X(:) Y(:)];

for jj = 1:50

    [mu,sig] = Z.eval(XX);

    ins(:,jj) = [mu(:);sig(:)]';

    outs(:,jj) = utils.softargmax(Decider.forward(ins(:,jj)'))';

    out = reshape(outs(:,jj),size(X));

    [~,indices(jj)] = max(outs(:,jj));

    dx = ((ub-lb)/20)'.*(rand(2,1)-0.5);

    xn = [X(indices(jj)) Y(indices(jj))] + dx';

    [x1,flag] = utils.catunique(x1,xn,1e-6);
    if flag
        y1 = [y1; ff(xn)];
    end

    yh1 = Z.eval(xn);

    Z = Z.condition(x1,y1,lb,ub);

    yh2 = Z.eval(xn);

    Ri(jj) = abs(yh2 - yh1);

    R2z(jj) = 1 - mean((yy - Z.eval_mu(xx)).^2)./var(yy);
    RMAEz(jj) = max(abs(yy - Z.eval_mu(xx)))./std(yy);
    RMSEz(jj) = mean((yy - Z.eval_mu(xx)).^2)./var(yy);
    if jj>1
        dRMSEz(jj) = RMSEz(jj-1) - RMSEz(jj);
    else
        dRMSEz(jj) = RMSEz0 - RMSEz(jj);
    end

    if plot
        figure(1)
        clf(1)
        utils.contourf(Z,(lb+ub)/2,1,2,'color','r')

        figure(2)
        clf(2)
        pcolor(out)
        shading flat

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

    end

    

    % if RMAEz(jj)<0.1
    %     break
    % end

end