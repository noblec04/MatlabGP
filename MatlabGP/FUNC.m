%{
    Gaussian Process
    
    An exact Gaussian Process with Gaussian Likelihood.

    A mean and kernel function can be created from which the GP can be
    generated.

    The GP can then be conditioned on training data.

    The GP can then be trained to optimize the HPs of the mean and kernel
    by finding the mean of the posterior distribution over parameters, or
    by finding the MAP estimate if the number of HPs is large.

%}

classdef FUNC
    
    properties
        f

        lb_x=0;
        ub_x=1;
    end

    methods

        function obj = FUNC(f,lb,ub)
            obj.f = f;
            obj.lb_x = lb;
            obj.ub_x = ub;
        end

        function [y,sig] = eval(obj,x)
            
            y = obj.f(x);

            if nargout>1
                sig = 0*x(:,1);
            end

        end
        
        function [y,dy] = eval_mu(obj,x)
            
            [a,b] = size(x);

            xAD = AutoDiff(x);

            fAD = obj.f(xAD);
            y = getvalue(fAD);

            if nargout>1
                J = full(getderivs(fAD));
                dy = -1*squeeze(reshape(J,[a b]));
            end

        end

        function [sig,dsig] = eval_var(~,x)
            
            sig = 0*x(:,1);

            if nargout>1
                dsig = 0*x;
            end

        end
    end
end