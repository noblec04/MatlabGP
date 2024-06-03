function [x,R,Q,Z] = opt(f,FF,lb,ub)

x0 = lb + (ub - lb).*lhsdesign(10,length(lb));
y0 = f(x0);

a = means.linear(ones(1,length(lb)))*means.linear(ones(1,length(lb))) + means.linear(ones(1,length(lb))) + means.const(1);

b = kernels.RQ(1,0.2*ones(1,length(lb)));
b.signn = 0.00001;

Z = GP(a,b);
Z = Z.condition(x0,y0,lb,ub);
Z.lb_x = lb;
Z.ub_x = ub;
Z = Z.train();

%fx = figure();

for i = 1:50
    i
    [x] = BO.argmin(FF,Z);

    y = f(x);

    x0 = [x0;x];
    y0 = [y0;y];

    %Z = Z.condition(x0,y0,lb,ub);
    Z = Z.resolve(x,y);
    Z.lb_x = lb;
    Z.ub_x = ub;

    % if i<10
    %     Z = Z.train();
    % end

    [Q(i)] = min(Z.Y);

    % figure(fx)
    % clf(fx)
    % utils.plotSurf(Z,1,2)
    % hold on
    % plot3(x(:,1),x(:,2),y,'x')
end

[R,i] = min(Z.Y);
x = Z.X(i,:);

end