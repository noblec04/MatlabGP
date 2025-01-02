function [f] = Kursawe(x)

% -5<x<5

f = 0*x(:,[1 2]);

for i = 1:2
    f(:,1) = f(:,1) + (-10*exp(-0.2*sqrt(x(:,i).^2 + x(:,i+1).^2)));
end

for i = 1:3
    f(:,2) = f(:,2) + (-10*exp(-0.2*sqrt(abs(x(:,i)).^(0.8) + 5*sin(x(:,i).^2))));
end

f = real(f);


end