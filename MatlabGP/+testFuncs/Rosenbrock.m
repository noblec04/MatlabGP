function [y] = Rosenbrock(x,i)

switch i
    case 1
        d = size(x,2);
        sum = 0;
        for ii = 1:(d-1)
        	xi = x(:,ii);
        	xnext = x(:,ii+1);
        	new = 100*(xnext-xi.^2).^2 + (xi-1).^2;
        	sum = sum + new;
        end

        y = sum;%/7210;
    case 2
        d = size(x,2);
        sum = 0;
        for ii = 1:(d-1)
        	xi = x(:,ii);
        	xnext = x(:,ii+1);
        	new = 50*(xnext-xi.^2).^2 + (-xi-2).^2;
            new2 = 0.5*xi;
        	sum = sum + new+ new2;
        end

        y = sum;%/7210;

    case 3
        y1 = testFuncs.Rosenbrock(x,1);%*7210;

        d = size(x,2);
        sum = 0;
        sum2= 0;
        for ii = 1:(d-1)
        	xi = x(:,ii);
            new = 0.5*xi;
        	sum = sum + new;

            x1 = x(:,1);
            new2 = 0.5*x1;
        	sum2 = sum2 + new2;
        end

        y = (y1 - 4 - sum)./(10 + sum2);%(1/7210)*
end



end