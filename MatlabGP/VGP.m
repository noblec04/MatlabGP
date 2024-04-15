classdef VGP
    
    properties
        kernel
        mean

        M
        Minv

        Kuf

        Kuu
        Kuuinv

        alpha
        signn

        X
        Xu
        Y

        lb_x
        ub_x
    end

    methods

        function obj = VGP(mean,kernel,ind)
            if isempty(mean)
                mean = @(x) 0*x(:,1);
            end
            obj.mean = mean;
            obj.kernel = kernel;
            obj.Xu = ind;
        end

        function [y,sig] = eval(obj,x)
            
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xu = (obj.Xu - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksu = obj.kernel.build(xs,xu);

            y = obj.mean(x) + ksu*obj.alpha;

            if nargout>1

                sigs = -dot(ksu',(obj.Kuuinv)*ksu') + dot(ksu',obj.Minv*ksu');
            
                sig = obj.kernel.scale^2 + obj.kernel.signn^2 + sigs;
            end

        end

        function y = samplePrior(obj,x)
            
            K = obj.kernel.build(x,x);

            y = mvnrnd(0*x(:,1),K);

        end

        function y = samplePosterior(obj,x)
            
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xu = (obj.Xu - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksu = obj.kernel.build(xs,xu);
            kss = obj.kernel.build(xs,xs);

            sigs = -ksu*obj.Kuuinv*ksu' + ksu*obj.Minv*ksu';
            
            sig = kss + obj.kernel.signn^2 + sigs;

            y = mvnrnd(ksu*obj.alpha,sig);
            
        end

        function obj = condition(obj,X,Y)

            obj.X = X;
            obj.Y = Y;

            obj.lb_x = min(X);
            obj.ub_x = max(X);

            xf = (X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xu = (obj.Xu - obj.lb_x)./(obj.ub_x - obj.lb_x);

            obj.kernel.scale = std(Y)/sqrt(2);

            obj.Kuu = obj.kernel.build(xu,xu);
            obj.Kuuinv = pinv(obj.Kuu,1*10^(-7));

            obj.Kuf = obj.kernel.build(xu,xf);

            obj.M = obj.Kuu + obj.Kuf*obj.Kuf'/obj.kernel.signn;
            obj.Minv = pinv(obj.M,1*10^(-7));

            obj.alpha = obj.Minv*obj.Kuf*Y/obj.kernel.signn;

        end

        function nll = LL(obj,theta,regress)

            if regress
                obj.kernel.signn = theta(end);
                theta(end) = [];
            end
            
            obj.kernel = obj.kernel.setHPs(theta);

            obj = obj.condition(obj.X,obj.Y);

            its = randsample(size(obj.X,1),max(5,ceil(size(obj.X,1)/50)));

            [mu,sig] = obj.eval(obj.X(its,:));
            
            nll = sum(-log(2*pi*sqrt(abs(sig'))) - ((obj.Y(its) - mu).^2)./sig') + 0.05*sum(log(gampdf(theta,2,0.5)));

            nll = -1*nll;
        end

        function [obj,nll] = train(obj,regress)

            if nargin<2
                regress=0;
            end
           
            tx0 = obj.kernel.getHPs();

            if regress
                tx0(end+1) = 0;
            end

            tlb = 0*tx0 + 0.01;
            tub = 0*tx0 + 8;

            func = @(x) obj.LL(x,regress);
            opts = bads('Defaults');
            opts.Display = 'final';
            opts.TolFun = 10^(-2);
            opts.TolMesh = 10^(-2);

            for i = 1:3
                tx0 = tlb + (tub - tlb).*rand(1,length(tlb));
                
                [theta{i},val(i)] = bads(func,tx0,tlb,tub);

            end

            [nll,i] = min(val);

            theta = theta{i};

            if regress
                obj.kernel.signn = theta(end);
                theta(end) = [];
            end

            obj.kernel = obj.kernel.setHPs(theta);
            obj = obj.condition(obj.X,obj.Y);

        end
        
    end
end