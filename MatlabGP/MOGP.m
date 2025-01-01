%{
    Multi-output Gaussian Process
    
    A set of single-task exact Gaussian Process with Gaussian Likelihood.

    A mean and kernel function can be created from which the GP can be
    generated.

    The GP can then be conditioned on training data.

    The GP can then be trained to optimize the HPs of the mean and kernel
    by finding the mean of the posterior distribution over parameters, or
    by finding the MAP estimate if the number of HPs is large.

%}

classdef MOGP
    
    properties
        GPs

        mean
        kernel

        X
        Y

        lb_x=0;
        ub_x=1;
    end

    methods

        function obj = MOGP(mean,kernel,N)
            if isempty(mean)
                mean = means.zero;
            end
            obj.mean = mean;
            obj.kernel = kernel;

            for i = 1:N
                obj.GPs{i} = GP(mean,kernel);
            end
        end

        function [y,sig] = eval(obj,x)
            
            nY = numel(obj.GPs);
            for i = 1:nY

                if nargout==2
                    [y(:,i),sig(:,i)] = obj.GPs{i}.eval(x);
                else
                    [y(:,i)] = obj.GPs{i}.eval(x);
                end

            end
        end
        
        function [y] = eval_mu(obj,x)
            
            nY = numel(obj.GPs);
            for i = 1:nY

                y(:,i) = obj.GPs{i}.eval_mu(x);

            end

        end

        function [sig] = eval_var(obj,x)
            
            nY = numel(obj.GPs);
            for i = 1:nY

                sig(:,i) = obj.GPs{i}.eval_var(x);

            end

        end

        function y = sample(obj,x)

            nY = numel(obj.GPs);
            for i = 1:nY

                y(:,i) = obj.GPs{i}.sample(x);

            end

        end

        function [dy,dsig] = eval_grad(obj,x)
            
            nY = numel(obj.GPs);
            for i = 1:nY

                [dy(:,i,:),dsig(:,i,:)] = obj.GPs{i}.eval_grad(x);

            end

            
        end

        function y = LCB(obj,x)

            [y,sig] = obj.eval(x);

            y = y - 2*sqrt(sig);

        end

        function y = UCB(obj,x)

            [y,sig] = obj.eval(x);

            y = y + 2*sqrt(sig);

        end

        function [obj] = condition(obj,X,Y,lb,ub)

            obj.X = X;
            obj.Y = Y;

            if nargin<4
                obj.lb_x = min(X);
                obj.ub_x = max(X);
            else
                obj.lb_x = lb;
                obj.ub_x = ub;
            end

            nY = numel(obj.GPs);
            for i = 1:nY

                obj.GPs{i} = obj.GPs{i}.condition(X,Y(:,i),obj.lb_x,obj.ub_x);

            end

        end

        function L = LOO(obj)

            nY = numel(obj.GPs);
            for i = 1:nY

                L(:,i) = obj.GPs{i}.LOO();

            end
        end

        function [theta] = getHPs(obj)

            nY = numel(obj.GPs);

            theta=[];

            for i = 1:nY

                theta = [theta; obj.GPs{i}.getHPs()];

            end

        end

        function obj = setHPs(obj,theta)

            nY = numel(obj.GPs);

            nn = 0;
            for i = 1:nY

                thetai = obj.GPs{i}.getHPs();
                nT = length(thetai);

                obj.GPs{i} = obj.GPs{i}.setHPs(theta(1+nn:nT+nn));

                nn = nn + nT;
            end

        end

        function [obj,LL] = train(obj,regress)

            if nargin<2
                regress=0;
            end

            nY = numel(obj.GPs);
            for i = 1:nY

                [obj.GPs{i},LL{i}] = obj.GPs{i}.train(regress);

            end
        end

        function [obj, LL] = train2(obj)

            nY = numel(obj.GPs);
            for i = 1:nY

                [obj.GPs{i},LL(i)] = obj.GPs{i}.train2();

            end
            
        end

        function obj = resolve(obj,x,y)
           
            replicates = ismembertol(x,obj.X,1e-4,'ByRows',true);

            x(replicates,:) = [];
            y(replicates,:) = [];

            if size(x,1)>0

                obj.X = [obj.X; x];
                obj.Y = [obj.Y; y];
           
                nY = numel(obj.GPs);
                for i = 1:nY

                    obj.GPs{i} = obj.GPs{i}.resolve(x,y(:,i));

                end
            end

        end

        %Operator Overloading


        function obj = or(obj,A)
            x = A(:,1:end-1);
            y = A(:,end);

            obj = obj.condition(x,y);
        end

        function warpedobj = exp(obj)

           warpedobj = warpGP(obj,'exp');

        end

        function warpedobj = cos(obj)

           warpedobj = warpGP(obj,'cos');

        end

        function warpedobj = sin(obj)

           warpedobj = warpGP(obj,'sin');

        end

        function warpedobj = mpower(obj,~)

           warpedobj = warpGP(obj,'square');

        end
        
    end
end