classdef GP
    
    properties
        kernel
        mean

        K
        Kinv
        alpha
        signn

        X
        Y

        lb_x
        ub_x
    end

    methods

        function obj = GP(mean,kernel)
            if isempty(mean)
                mean = @(x) 0*x(:,1);
            end
            obj.mean = mean;
            obj.kernel = kernel;
        end

        function [y,sig] = eval(obj,x)
            
            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksf = obj.kernel.build(xs,xx);

            y = obj.mean(x) + ksf*obj.alpha;

            if nargout>1
                kss = obj.kernel.build(xs,xs);
                sig = abs(diag(kss) - dot(ksf',obj.Kinv*ksf')');
            end

        end

        function y = samplePrior(obj,x)
            
            K = obj.kernel.build(x,x);

            y = mvnrnd(0*x(:,1),K);

        end

        function y = samplePosterior(obj,x)
            
            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksf = obj.kernel.build(xs,xx);

            kss = obj.kernel.build(xs,xs);

            sig = kss - ksf*obj.Kinv*ksf' + 5*eps*eye(size(x,1));

            y = mvnrnd(ksf*obj.alpha,sig);
            
        end

        function obj = condition(obj,X,Y)

            obj.X = X;
            obj.Y = Y;

            obj.lb_x = min(X);
            obj.ub_x = max(X);

            xx = (X - obj.lb_x)./(obj.ub_x - obj.lb_x);

            obj.K = obj.kernel.build(xx,xx)+diag(0*xx+obj.kernel.signn);
            obj.Kinv = pinv(obj.K,1*10^(-7));

            obj.alpha = obj.Kinv*(obj.Y - obj.mean(obj.X));

        end

        function nll = LL(obj,theta,regress)

            if regress
                obj.kernel.signn = theta(end);
                theta(end) = [];
            end
            
            obj.kernel = obj.kernel.setHPs(theta);

            obj = obj.condition(obj.X,obj.Y);

            res = obj.Y - obj.mean(obj.X);

            detk = det(obj.K);

            if isnan(detk)
                detk = eps;
            end

            nll = -0.5*(eps + abs((res)'*obj.Kinv*(res))) - 0.5*log(abs(detk)+eps) + 5*sum(log(gampdf(theta,3,0.5)));

        end

        function obj = train(obj,regress)

            if nargin<2
                regress=0;
            end
           
            tx0 = obj.kernel.getHPs();

            if regress
                tx0(end+1) = 0;
            end

            tlb = 0*tx0 + 0.01;
            tub = 0*tx0 + 8;

            func = @(x) -1*obj.LL(x,regress);
            options = optimoptions('fmincon','Display','none');


            for i = 1:5
                tx0 = tlb + (tub - tlb).*rand(1,length(tlb));

                [theta{i},val(i)] = fmincon(func,tx0,[],[],[],[],tlb,tub,[],options);

            end

            [~,i] = min(val);

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