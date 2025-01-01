
clear
close all
clc

xx = [lhsdesign(10,2)];
yy = normrnd(forr(xx,0),0*forr(xx,0));

xmesh = lhsdesign(500,2);
ymesh = forr(xmesh,0);

ma = means.const(1);
ka = kernels.Matern32(1,[1 1]);
ka.signn = 1e-10;

Z = MOGP(ma,ka,3);
Z = Z.condition(xx,yy,[0 0],[1 1]);
Z = Z.train();

utils.plotSurf(Z,[0 0],1,2)
hold on
plot3(xx(:,2),xx(:,1),yy,'x')
plot3(xmesh(:,2),xmesh(:,1),ymesh,'.')

%%
for i = 1:20

    xm = lhsdesign(200,2);
    YN = Z.UCB(xm);
    An = utils.ParetoFront(YN);

    xn = xm(An==1,:);
    if size(xn,1)>=5
        xn = xn(1:5,:);
    end
    yn = normrnd(forr(xn,0),0*forr(xn,0));%forr(xn,0);

    xx = [xx;xn];
    yy = [yy;yn];

    Z = Z.condition(xx,yy,[0 0],[1 1]);

    Az = utils.ParetoFront(Z.Y);
    
    figure(2)
    clf(2)
    plot3(Z.Y(:,1),Z.Y(:,2),Z.Y(:,3),'.')
    hold on
    plot3(Z.Y(Az==1,1),Z.Y(Az==1,2),Z.Y(Az==1,3),'x')
    plot3(YN(An==1,1),YN(An==1,2),YN(An==1,3),'+')
    drawnow
end

%%

function y = forr(x,dx)

nx = size(x,1);

A = 0.5; B = 10; C = -5;

for i = 1:nx
    if x(i)<0.45
        y(i,1) = (6*x(i,1)-2).^2.*sin(12*x(i,2)-4);
    else
        y(i,1) = (6*x(i,1)-2).^2.*sin(12*x(i,2)-4)+dx;
    end

    y(i,2) = 0.4*(6*x(i,1)-2).^2.*sin(12*x(i,2)-4)-x(i,1)-1;
    y(i,3) = A*(6*x(i,1)-2).^2.*sin(12*x(i,2)-4)+B*(x(i,2)-0.5)-C;
end

end

