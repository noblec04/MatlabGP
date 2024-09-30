function [y] = Rosenbrock_image(xx,r,sig)

if nargin<3
    sig=0.01;
end

x1 = linspace(-2,2,28);
x2 = linspace(-2,2,28);

[X1,X2,X3] = meshgrid(x1,x2,xx);

xxx = [X1(:) X2(:) X3(:)];

d = size(xxx,2);
sum = 0;
for ii = 1:(d-1)
	xi = xxx(:,ii);
	xnext = xxx(:,ii+1);
	new = 100*(xnext-xi.^2).^2 + (xi-1).^2;
	sum = sum + new;
end

y = reshape(sum,[28 28 length(xx)])/7210;

if r
    y = y + normrnd(0*y,0*y+sig);
end

end
