function [XX,YY] = AdaptiveGridSampling(func,lb,ub,nL,nS,nT)

xn = lb + (ub-lb).*lhsdesign(nS,length(lb));

ran = ub-lb;

XX = xn;

YY = [];

for ii = 1:nL

    for jj = 1:size(xn,1)
        yy(jj) = func(xn(jj,:));
        YY = [YY;yy(jj)];
    end

    [~,ib] = sort(yy,'descend');

    xi = [];
    for kk = 1:nT

        lbi = max(lb,xn(ib(kk),:)-ran/(3*ii));
        ubi = min(ub,xn(ib(kk),:)+ran/(3*ii));

        xi = [xi;lbi + (ubi-lbi).*lhsdesign(nS,length(lb))];
    end

    xn = xi;

    XX = [XX;xn];

end

for jj = 1:size(xn,1)
    yy(jj) = func(xn(jj,:));
    YY = [YY;yy(jj)];
end

end