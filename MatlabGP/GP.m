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
        
        function [y,dy] = eval_mu(obj,x)
            
            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksf = obj.kernel.build(xs,xx);

            y = obj.mean(x) + ksf*obj.alpha;

            if nargout>1
                ksf = obj.kernel.grad(xs,xx);
                [~,dm] = obj.mean(x);

                dy = dm + ksf*obj.alpha;
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

        function [obj,dK] = condition(obj,X,Y)

            obj.X = X;
            obj.Y = Y;

            obj.lb_x = min(X);
            obj.ub_x = max(X);

            xx = (X - obj.lb_x)./(obj.ub_x - obj.lb_x);

            [obj.K] = obj.kernel.build(xx,xx);

            res = obj.Y - obj.mean(obj.X);

            kkp = pinv(obj.K,1*10^(-7));

            sigp = sqrt(dot((res'),kkp*(res))./(size(obj.Y,1)));
            
            if isinf(sigp)
                sigp = std(obj.Y);
            end

            obj.kernel.scale = sigp^2;

            if nargout==2
                [obj.K,dK] = obj.kernel.build(xx,xx);
            else
                [obj.K] = obj.kernel.build(xx,xx);
            end

            obj.K = obj.K + diag(0*xx+obj.kernel.signn);

            obj.Kinv = pinv(obj.K,1*10^(-7));

            obj.alpha = obj.Kinv*(res);

        end

        function [nll,dnLL] = LL(obj,theta,regress)

            if regress
                obj.kernel.signn = theta(end);
                theta(end) = [];
            end
            
            obj.kernel = obj.kernel.setHPs(theta);
            
            if nargout == 2
                [obj,dK] = obj.condition(obj.X,obj.Y);
            else
                [obj] = obj.condition(obj.X,obj.Y);
            end

            detk = det(obj.K);

            if isnan(detk)
                detk = eps;
            end

            nll = -0.5*log(sqrt(obj.kernel.scale)) - 0.5*log(abs(detk)+eps);% + 1*sum(log(gampdf(theta,3,2)));

            nll = -1*nll;

            if nargout==2
                dnLL = zeros(1,length(theta));
                for i = 1:length(theta)
                    dnLL(i) = -0.5*sum(sum((obj.alpha*obj.alpha' - obj.Kinv)*squeeze(dK(:,:,i)))) + (3 - 1*theta(i) - 1)/(length(theta)*theta(i)) + 1.4;
                end

                if regress
                    dnLL(end+1) = -1*sum(sum(2*sqrt(obj.kernel.signn)*(obj.alpha*obj.alpha' - obj.Kinv)));
                end
            end

        end

        function [obj,mval] = train(obj,regress)

            if nargin<2
                regress=0;
            end
           
            tx0 = obj.kernel.getHPs();

            tlb = 0*tx0 + 0.01;
            tub = 0*tx0 + 3;

            if regress
                tlb(end+1) = 0;
                tub(end+1) = 5;
            end

            func = @(x) obj.LL(x,regress);

            for i = 1:5
                tx0 = tlb + (tub - tlb).*rand(1,length(tlb));
                                
                [theta{i},val(i)] = bads(func,tx0,tlb,tub);
                %[theta{i},val(i)] = VSGD(func,tx0,'lr',0.1,'lb',tlb,'ub',tub,'gamma',0.01,'iters',400,'tol',1*10^(-3));

            end

            [mval,i] = min(val);

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