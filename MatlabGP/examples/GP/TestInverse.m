clear all
clc

ff = @(x) (6*x(:,3)-2).^2.*sin(x(:,1).*x(:,3)-x(:,2));

lb = [11 3 0];
ub = [13 5 1];

x1 = lb + (ub - lb).*lhsdesign(100,3);
y1 = ff(x1);

x = linspace(0,1,200);

ma = means.linear([1 1 1]);
ka = kernels.EQ(1,0.2);
ka.signn = 0.1;

Z = GP(ma,ka);
Z = Z.condition(x1,y1);
Z = Z.train();

Ftarget = 4;

%%

post = posterior(Ftarget,@(x) forr(Z,x), @(x) 0*x + 1);

pp = post(x);

%%

function y = forr(Z,x)

for j = 1:1000

    A = normrnd(0*x + 12,1);
    B = normrnd(0*x + 4,0.5);

    y(j,:) = Z.sample([A(:) B(:) x(:)]);
    %y(j,:) = (6*x-2).^2.*sin(A*x-B) + normrnd(0,0*x+3);
end

end

function P = posterior(target,func,prior)

mu = @(x) mean(func(x));
sig = @(x) std(func(x));

LL =@(x) -0.5*log(2*pi*sig(x).^2) - ((target - mu(x)).^2)./sig(x).^2 + log(prior(x));

P = @(x) exp(LL(x))/trapz(x,exp(LL(x)));

end