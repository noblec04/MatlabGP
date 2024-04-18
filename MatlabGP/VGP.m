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
                mean = means.zero;
            end
            obj.mean = mean;
            obj.kernel = kernel;
            obj.Xu = ind;
        end

        function [y,sig] = eval(obj,x)
            
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xu = (obj.Xu - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksu = obj.kernel.build(xs,xu);

            y = obj.mean.eval(x) + ksu*obj.alpha;

            if nargout>1

                kss = obj.kernel.build(xs,xs);

                sigs = -dot(ksu',(obj.Kuuinv)*ksu') + dot(ksu',obj.Minv*ksu');
            
                sig = diag(kss) + obj.kernel.signn + sigs';
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
            
            sig = kss + obj.kernel.signn + sigs;

            y = mvnrnd(obj.mean.eval(x)+ksu*obj.alpha,sig);
            
        end

        function obj = condition(obj,X,Y)

            obj.X = X;
            obj.Y = Y;

            obj.lb_x = min(X);
            obj.ub_x = max(X);

            xf = (X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xu = (obj.Xu - obj.lb_x)./(obj.ub_x - obj.lb_x);

            obj.kernel.scale = std(Y)/2;

            obj.Kuu = obj.kernel.build(xu,xu);
            obj.Kuuinv = inv(obj.Kuu);

            obj.Kuf = obj.kernel.build(xu,xf);

            obj.M = obj.Kuu + obj.Kuf*obj.Kuf'/obj.kernel.signn;
            obj.Minv = inv(obj.M);

            obj.alpha = obj.Minv*obj.Kuf*(Y - obj.mean.eval(X))/obj.kernel.signn;

        end

        function nll = LL(obj,theta,regress,ntm)

            if regress
                obj.kernel.signn = theta(end);
                theta(end) = [];
            end
            
            obj.mean = obj.mean.setHPs(theta(1:ntm));
            obj.kernel = obj.kernel.setHPs(theta(ntm+1:end));

            obj = obj.condition(obj.X,obj.Y);

            its = randsample(size(obj.X,1),max(5,ceil(size(obj.X,1)/50)));

            [mu,sig] = obj.eval(obj.X(its,:));
            
            nll = sum(-log(2*pi*sqrt(abs(sig))) - ((obj.Y(its) - mu).^2)./sig) + 0.05*sum(log(gampdf(abs(theta)+eps,2,0.5)));

            nll = -1*nll;

            nll(isnan(nll)) = 0;
            nll(isinf(nll)) = 0;
        end

        function [obj,nll] = train(obj,regress)

            if nargin<2
                regress=1;
            end
           
            tm0 = obj.mean.getHPs();
            ntm = numel(tm0);

            tmlb = 0*tm0 - 10;
            tmub = 0*tm0 + 10;

            tk0 = obj.kernel.getHPs();

            tklb = 0*tk0 + 0.01;
            tkub = 0*tk0 + 3;

            tlb = [tmlb tklb];
            tub = [tmub tkub];

            if regress
                tlb(end+1) = 0;
                tub(end+1) = 5;
            end

            func = @(x) obj.LL(x,regress,ntm);
            opts = bads('Defaults');
            opts.Display = 'final';
            opts.TolFun = 10^(-2);
            opts.TolMesh = 10^(-2);

            for i = 1:3
                tx0 = tlb + (tub - tlb).*rand(1,length(tlb));
                
                [theta{i},val(i)] = bads(func,tx0,tlb,tub,tlb,tub,[],opts);

            end

            [nll,i] = min(val);

            theta = theta{i};

            if regress
                obj.kernel.signn = theta(end);
                theta(end) = [];
            end

            obj.mean = obj.mean.setHPs(theta(1:ntm));
            obj.kernel = obj.kernel.setHPs(theta(ntm+1:end));
            obj = obj.condition(obj.X,obj.Y);

        end
        
    end
end