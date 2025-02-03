function [y] = rosen_image_2D(xx,sig)

x1 = linspace(-2,2,28);
x2 = linspace(-2,2,28);

[X1,X2] = ndgrid(x1,x2);

xxx = [];

for i = 1:size(xx,1)
    xxx = [xxx; [0*X1(:)+xx(i,1) 0*X1(:)+xx(i,2) X1(:) X2(:)]];
end

d = size(xxx,2);
sum = 0;
for ii = 1:(d-1)
	xi = xxx(:,ii);
	xnext = xxx(:,ii+1);
	new = 50*(xnext-xi.^2).^2 + (-xi-2).^2;
    new2 = 0.5*xi;
	sum = sum + new+ new2;

end

y = reshape(sum,[size(xx,1) 28 28])/7210;

if sig~=0
    y = normrnd(y,0*y+sig);
end

end
