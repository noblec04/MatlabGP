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

        lb_x=0;
        ub_x=1;
    end

    methods

        function obj = GP(mean,kernel)
            if isempty(mean)
                mean = means.zero;
            end
            obj.mean = mean;
            obj.kernel = kernel;
        end

        function [y,sig] = eval(obj,x)
            
            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksf = obj.kernel.build(xs,xx);

            y = obj.mean.eval(x) + ksf*obj.alpha;

            if nargout>1
                kss = obj.kernel.build(xs,xs);
                sig = abs(diag(kss) - dot(ksf',obj.Kinv*ksf')');
            end

        end
        
        function [y,dy] = eval_mu(obj,x)
            
            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksf = obj.kernel.build(xs,xx);

            y = obj.mean.eval(x) + ksf*obj.alpha;

            if nargout>1
                dksf = obj.kernel.grad(xs,xx);
                [~,dm] = obj.mean.eval(x);

                dy = dm + dksf*obj.alpha;
            end

        end

        function [sig,dsig] = eval_var(obj,x)
            
            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksf = obj.kernel.build(xs,xx);

            kss = obj.kernel.build(xs,xs);
            sig = -1*abs(diag(kss) - dot(ksf',obj.Kinv*ksf')');

            if nargout>1
                dksf = obj.kernel.grad(xs,xx);
                dsig = -1*(-2*dot(ksf,obj.Kinv*dksf')');
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

            y = mvnrnd(obj.mean.eval(x) + ksf*obj.alpha,sig);
            
        end

        function [obj,dm,dK] = condition(obj,X,Y)

            obj.X = X;
            obj.Y = Y;

            obj.lb_x = min(X);
            obj.ub_x = max(X);

            xx = (X - obj.lb_x)./(obj.ub_x - obj.lb_x);

            [obj.K] = obj.kernel.build(xx,xx);
            
            if nargout>1
                [mm,dm] = obj.mean.eval(obj.X);

                res = obj.Y - mm;
            else
                res = obj.Y - obj.mean.eval(obj.X);
            end

            kkp = pinv(obj.K,1*10^(-7));

            sigp = sqrt(dot((res'),kkp*(res))./(size(obj.Y,1)));
            
            if isinf(sigp)
                sigp = std(obj.Y);
            end

            obj.kernel.scale = sigp^2;

            if nargout>1
                [obj.K,dK] = obj.kernel.build(xx,xx);
            else
                [obj.K] = obj.kernel.build(xx,xx);
            end

            obj.K = obj.K + diag(0*xx+obj.kernel.signn);

            obj.Kinv = pinv(obj.K,1*10^(-7));

            obj.alpha = obj.Kinv*(res);

        end

        function [nll,dnLL] = LL(obj,theta,regress,ntm)

            if regress
                obj.kernel.signn = theta(end);
                theta(end) = [];
            end
            
            obj.mean = obj.mean.setHPs(theta(1:ntm));
            obj.kernel = obj.kernel.setHPs(theta(ntm+1:end));
            
            if nargout == 2
                [obj,dm,dK] = obj.condition(obj.X,obj.Y);
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

                for i = 1:ntm
                    dnLL(i) = 0;
                end
                n=0;
                for i = ntm+1:length(theta)
                    n=n+1;
                    dnLL(i) = -0.5*sum(sum((obj.alpha*obj.alpha' - obj.Kinv)*squeeze(dK(:,:,n)))) + (3 - 1*theta(n) - 1)/(length(theta)*theta(n)) + 1.4;
                end

                if regress
                    dnLL(end+1) = -1*sum(sum(2*sqrt(obj.kernel.signn)*(obj.alpha*obj.alpha' - obj.Kinv)));
                end
            end

        end

        function [obj,LL] = train(obj,regress)

            if nargin<2
                regress=0;
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

            xxt = tlb + (tub - tlb).*lhsdesign(100*length(tlb),length(tlb));

            for ii = 1:size(xxt,1)
                LL(ii) = -1*func(xxt(ii,:));
            end

            LL = exp(1 + LL - max(LL));

            theta = sum(xxt.*LL')/sum(LL);

            % for i = 1:5
            %     tx0 = tlb + (tub - tlb).*rand(1,length(tlb));
            % 
            %     %[theta{i},val(i)] = bads(func,tx0,tlb,tub);
            %     [theta{i},val(i)] = VSGD(func,tx0,'lr',0.1,'lb',tlb,'ub',tub,'gamma',0.01,'iters',400,'tol',1*10^(-3));
            % 
            % end
            % 
            % [mval,i] = min(val);
            % 
            % theta = theta{i};

            if regress
                obj.kernel.signn = theta(end);
                theta(end) = [];
            end


            obj.mean = obj.mean.setHPs(theta(1:ntm));
            obj.kernel = obj.kernel.setHPs(theta(ntm+1:end));
            obj = obj.condition(obj.X,obj.Y);

        end

        function obj = resolve(obj,x,y)
           
            replicates = ismembertol(x,obj.X,1e-4,'ByRows',true);

            x(replicates,:) = [];
            y(replicates,:) = [];

            if size(x,1)>0
               
                %{
                    Using single point partitioned matrix inverse equation
                    from Binois et. al. (2019) https://doi.org/10.1080/00401706.2018.1469433
                    with ~500x500 matrix ~10x speedup
                %}
               
                xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
                xsc = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
               
                [ks2] = obj.kernel.build(xsc,xx);
               
                c2 = obj.kernel.scale - dot(ks2,obj.Kinv*ks2');
               
                g = -1*obj.Kinv*ks2'/c2;
               
                gg = g*g';
               
                k11inv1 = obj.Kinv + (c2)*gg;
               
                k11inv2 = [k11inv1 g;
                            g' 1/(c2)];
                       
                obj.Kinv = k11inv2;
               
                obj.X = [obj.X; x];
                obj.Y = [obj.Y; y];
           
                obj.alpha = obj.Kinv*(obj.Y - obj.mean.eval(obj.X));
            end

        end


        function obj = or(obj,A)
            x = A(:,1:end-1);
            y = A(:,end);

            obj = obj.condition(x,y);
        end
        
    end
end