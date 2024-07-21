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
                sig = abs(diag(kss)  + obj.kernel.signn - dot(ksf',obj.Kinv*ksf')');
            end

        end
        
        function [y,dy] = eval_mu(obj,x)
            
            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksf = obj.kernel.build(xs,xx);

            y = obj.mean.eval(x) + ksf*obj.alpha;

            if nargout>1
                dksf = obj.kernel.grad(xs,xx);
                dm = obj.mean.grad(xs);

                dy = dm + (dksf*obj.alpha);
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
                dsig = -1*(-2*(ksf*obj.Kinv*dksf'));
            end

        end


        function y = samplePrior(obj,x)
            
            K = obj.kernel.build(x,x) + 5*eps*eye(size(x,1)) + diag(0*x(:,1)+obj.kernel.signn);

            y = mvnrnd(0*x(:,1),K);

        end

        function y = samplePosterior(obj,x)
            
            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksf = obj.kernel.build(xs,xx);

            kss = obj.kernel.build(xs,xs);

            sig = kss + diag(0*x(:,1)+obj.kernel.signn) - ksf*obj.Kinv*ksf' + 5*eps*eye(size(x,1));

            y = mvnrnd(obj.mean.eval(x) + ksf*obj.alpha,sig);
            
        end

        function [obj,dm,dK] = condition(obj,X,Y,lb,ub)

            obj.X = X;
            obj.Y = Y;

            if nargin<4
                obj.lb_x = min(X);
                obj.ub_x = max(X);
            else
                obj.lb_x = lb;
                obj.ub_x = ub;
            end

            xx = (X - obj.lb_x)./(obj.ub_x - obj.lb_x);

            obj.kernel.scale = 1;
            [obj.K] = obj.kernel.build(xx,xx);
            
            if nargout>1
                [mm,dm] = obj.mean.eval(obj.X);
                res = obj.Y - mm;
            else
                res = obj.Y - obj.mean.eval(obj.X);
            end

            kkp = pinv(obj.K,0);

            sigp = sqrt(abs(res'*kkp*res./(size(obj.Y,1))));
            %sigp = std(res);
            % if isinf(sigp)
            %     sigp = std(obj.Y);
            % end

            obj.kernel.scale = sigp^2;

            if nargout>1
                [obj.K,dK] = obj.kernel.build(xx,xx);
            else
                [obj.K] = obj.kernel.build(xx,xx);
            end

            obj.K = obj.K + diag(0*xx(:,1)+obj.kernel.signn);

            obj.Kinv = pinv(obj.K,0);

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

            detk = det(obj.K + diag(0*obj.K(:,1) + obj.kernel.signn));

            if isnan(detk)
                detk = eps;
            end

            nll = -0.5*log(sqrt(obj.kernel.scale)) - 0.5*log(abs(detk)+eps) + 0.1*sum(log(eps+gampdf(theta,1.1,0.5)));

            nll = nll;

            if nargout==2
                dnLL = zeros(1,length(theta));

                for i = 1:ntm
                    dnLL(i) = -2*obj.alpha'*dm(:,i);
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

        function [loss,dloss] = loss(obj,theta)

            tm0 = obj.mean.getHPs();
            ntm = numel(tm0);

            nV = length(theta(:));

            theta = AutoDiff(theta);

            obj = obj.setHPs(theta);

            [obj] = obj.condition(obj.X,obj.Y);

            detk = det(obj.K + diag(0*obj.K(:,1) + obj.kernel.signn));

            loss_nll = -0.5*log(sqrt(obj.kernel.scale)) - 0.5*log(abs(detk)+eps) + 0.1*sum(log(eps+gampdf(abs(theta(ntm+1:end)),3,5)));

            loss_nll = -1*loss_nll;

            loss = getvalue(loss_nll);
            dloss = getderivs(loss_nll);
            dloss = reshape(full(dloss),[1 nV]);

        end

        function [thetas,ntm,ntk,tm0,tk0] = getHPs(obj)

            tm0 = obj.mean.getHPs();
            tk0 = obj.kernel.getHPs();

            ntm = numel(tm0);
            ntk = numel(tk0);

            thetas = [tm0 tk0];

        end

        function obj = setHPs(obj,theta)

            [~,ntm,~] = obj.getHPs();
            
            obj.mean = obj.mean.setHPs(theta(1:ntm));
            obj.kernel = obj.kernel.setHPs(theta(ntm+1:end));

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

            tklb = 0*tk0 + 0.001;
            tkub = 0*tk0 + 10;

            tlb = [tmlb tklb];
            tub = [tmub tkub];
            %tx0 = [tm0 tk0];

            if regress
                tlb(end+1) = 0;
                tub(end+1) = 5;
            end

            func = @(x) obj.LL(x,regress,ntm);

            % xxt = tlb + (tub - tlb).*lhsdesign(200*length(tlb),length(tlb));
            % 
            % for ii = 1:size(xxt,1)
            %     LL(ii) = func(xxt(ii,:));
            % end
            % 
            % LL = exp(1 + LL - max(LL));
            % 
            % theta = sum(xxt.*LL')/sum(LL);

            for i = 1:2
                %tx0 = tlb + (tub - tlb).*rand(1,length(tlb));

                %opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'Display','none');

                %[theta{i},val(i)] = fmincon(func,tx0,[],[],[],[],tlb,tub,[],opts);

                %opts.TolMesh = 1e-2;
                %opts.TolFun = 1e-2;
                [theta{i},val(i)] = optim.RecursiveGrid(func,6,20,tlb,tub);

                %[theta{i},val(i)] = bads(func,tx0,tlb,tub,[],[],[],opts);
                %[theta{i},val(i)] = VSGD(func,tx0,'lr',0.02,'lb',tlb,'ub',tub,'gamma',0.0001,'iters',20,'tol',1*10^(-4));
                %[theta{i},val(i)] = optim.minimizebnd(func,tx0,tlb,tub,1,0);
            end

            [mval,i] = min(val)

            theta = theta{i};

            if regress
                obj.kernel.signn = theta(end);
                theta(end) = [];
            end


            obj.mean = obj.mean.setHPs(theta(1:ntm));
            obj.kernel = obj.kernel.setHPs(theta(ntm+1:end));
            obj = obj.condition(obj.X,obj.Y);

        end

        function [obj, LL] = train2(obj)


            tm0 = obj.mean.getHPs();
            ntm = numel(tm0);

            tmlb = 0*tm0 - 30;
            tmub = 0*tm0 + 30;

            tk0 = obj.kernel.getHPs();

            tklb = 0*tk0 + 0.0001;
            tkub = 0*tk0 + 30;

            tlb = [tmlb tklb];
            tub = [tmub tkub];

            func = @(x) obj.loss(x);

            for i = 1:3
                tx0 = tlb + (tub - tlb).*rand(1,length(tlb));

                opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'Display','off');

                [theta{i},val(i)] = fmincon(func,tx0,[],[],[],[],tlb,tub,[],opts);

                %[theta{i},val(i)] = VSGD(func,tx0,'lr',0.02,'lb',tlb,'ub',tub,'gamma',0.0001,'iters',20,'tol',1*10^(-4));

            end

            [LL,i] = min(val);

            theta = theta{i};

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