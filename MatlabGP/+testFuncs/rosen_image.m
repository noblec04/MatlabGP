function [y] = rosen_image(xx,sig)

x1 = linspace(-2,2,28)';
x2 = linspace(-2,2,28)';

[xxx] = utils.ndgrid(xx,x1,x2);

d = size(xxx,2);
sum = 0;
for ii = 1:(d-1)
	xi = xxx(:,ii);
	xnext = xxx(:,ii+1);
	new = 100*(xnext-xi.^2).^2 + (xi-1).^2;
	sum = sum + new;
end

y = reshape(sum,[length(xx) 28 28])/7210;

if sig~=0
    y = normrnd(y,0*y+sig);
end

end
