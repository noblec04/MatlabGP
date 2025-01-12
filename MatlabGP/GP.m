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
        L
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

                v = dot(ksf',(obj.K\ksf'))';
                sig = (abs(obj.kernel.scale - v));

                %sig = abs(obj.kernel.scale  + obj.kernel.signn - dot(ksf',obj.Kinv*ksf')');
            end
        end
        
        function [y] = eval_mu(obj,x)
            
            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksf = obj.kernel.build(xs,xx);

            y = obj.mean.eval(x) + ksf*obj.alpha;

        end

        function [sig] = eval_var(obj,x)
            
            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksf = obj.kernel.build(xs,xx);

            % v = obj.L\ksf';
            % sig = abs(obj.kernel.scale - sum(v.^2,1))';

            v = dot(ksf',(obj.K\ksf'));
            sig = abs(obj.kernel.scale - v);

            %sig = abs(obj.kernel.scale - dot(ksf',obj.Kinv*ksf')');

        end

        function y = sample(obj,x)
            [mu,sig] = obj.eval(x);

            y = normrnd(mu,sqrt(sig));
        end

        function [dy,dsig] = eval_grad(obj,x)
            
            [nn,nx] = size(x);

            x = AutoDiff(x);

            y = obj.eval_mu(x);

            dy = reshape(nonzeros(getderivs(y)),[nn,nx]);

            if nargout>1
                sig = obj.eval_var(x);

                dsig = reshape(nonzeros(getderivs(sig)),[nn,nx]);
            end

        end

        function y = samplePrior(obj,x)
            
            Kxx = obj.kernel.build(x,x) + 5*eps*eye(size(x,1)) + diag(0*x(:,1)+obj.kernel.signn);

            L1 =chol(Kxx+(1e-6)*eye(size(Kxx)));

            z = mvnrnd(0*x(:,1),eye(size(Kxx)));

            y = L1*z';

        end

        function y = samplePosterior(obj,x)
            
            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksf = obj.kernel.build(xs,xx);

            kss = obj.kernel.build(xs,xs);

            sig = kss + diag(0*x(:,1)+obj.kernel.signn) - ksf*obj.Kinv*ksf' + 5*eps*eye(size(x,1));

            y = mvnrnd(obj.mean.eval(x) + ksf*obj.alpha,sig);
            
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

            xx = (X - obj.lb_x)./(obj.ub_x - obj.lb_x);

            obj.kernel.scale = 1;
            [obj.K] = obj.kernel.build(xx,xx);
            obj.K = obj.K + diag(0*xx(:,1)+obj.kernel.signn) + (1e-14)*eye(size(xx,1));

            res = obj.Y - obj.mean.eval(obj.X);

            obj.alpha = obj.K\res;

            sigp = sqrt(abs(res'*obj.alpha./(size(obj.Y,1))));

            obj.kernel.scale = sigp^2;

            obj.K = obj.kernel.scale*obj.K;
            obj.alpha = obj.alpha/sigp^2;

        end

        function [nll] = LL(obj,theta,regress,ntm)

            if regress
                obj.kernel.signn = theta(end);
                theta(end) = [];
            end
            
            obj.mean = obj.mean.setHPs(theta(1:ntm));
            obj.kernel = obj.kernel.setHPs(theta(ntm+1:end));

            [obj] = obj.condition(obj.X,obj.Y);

            detk = det(obj.K + diag(0*obj.K(:,1) + obj.kernel.signn));

            if isnan(detk)||isinf(detk)
                detk = eps;
            end

            %nll = sum(obj.LOO());

            nll = -0.5*log(sqrt(obj.kernel.scale)) - 0.5*log(abs(detk)+eps) + 0.1*sum(log(eps+gampdf(abs(theta(ntm+1:end)),1.1,0.5)));

        end

        function [loss,dloss] = loss(obj,theta)

            tm0 = obj.mean.getHPs();
            ntm = numel(tm0);

            nV = length(theta(:));

            theta = AutoDiff(theta);

            obj = obj.setHPs(theta);

            [obj] = obj.condition(obj.X,obj.Y);

            detk = det(obj.K/obj.kernel.scale + diag(0*obj.K(:,1) + obj.kernel.signn));

            loss_nll = -0.5*log(sqrt(obj.kernel.scale)) - 0.5*log(abs(detk)+eps) + 1*sum(log(eps+gampdf(abs(theta(ntm+1:end)),1.1,0.5)));

            loss_nll = -1*loss_nll;

            loss = getvalue(loss_nll);
            dloss = getderivs(loss_nll);
            dloss = reshape(full(dloss),[1 nV]);

        end

        function L = LOO(obj)

            if isempty(obj.Kinv)
                obj.Kinv = obj.K\eye(size(obj.K,1));
            end
            L = 0.5*(obj.alpha.^2)./diag(obj.Kinv);% - 0.5*log(diag(obj.Kinv));
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

            [xxt,LL1] = optim.AdaptiveGridSampling(func,tlb,tub,10,20,4);

            % xxt = tlb + (tub - tlb).*lhsdesign(200*length(tlb),length(tlb));
            % 
            % for ii = 1:size(xxt,1)
            %     LL(ii) = func(xxt(ii,:));
            % end

            LL = exp(1 + LL1 - max(LL1));

            theta = sum(xxt.*LL(:))/sum(LL);

            % for i = 1:2
            %     %tx0 = tlb + (tub - tlb).*rand(1,length(tlb));
            % 
            %     %opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'Display','none');
            % 
            %     %[theta{i},val(i)] = fmincon(func,tx0,[],[],[],[],tlb,tub,[],opts);
            % 
            %     %opts.TolMesh = 1e-2;
            %     %opts.TolFun = 1e-2;
            %     [theta{i},val(i)] = optim.RecursiveGrid(func,6,20,tlb,tub);
            % 
            %     %[theta{i},val(i)] = bads(func,tx0,tlb,tub,[],[],[],opts);
            %     %[theta{i},val(i)] = VSGD(func,tx0,'lr',0.02,'lb',tlb,'ub',tub,'gamma',0.0001,'iters',20,'tol',1*10^(-4));
            %     %[theta{i},val(i)] = optim.minimizebnd(func,tx0,tlb,tub,1,0);
            % end
            % 
            % [mval,i] = min(val)
            % 
            % theta = theta{i};

            if regress
                obj.kernel.signn = theta(end);
                theta(end) = [];
            end

            obj.mean = obj.mean.setHPs(theta(1:ntm));
            obj.kernel = obj.kernel.setHPs(theta(ntm+1:end));
            obj = obj.condition(obj.X,obj.Y,obj.lb_x,obj.ub_x);

        end

        function [obj, LL] = train2(obj)


            tm0 = obj.mean.getHPs();
            ntm = numel(tm0);

            tmlb = 0*tm0 - 30;
            tmub = 0*tm0 + 30;

            tk0 = obj.kernel.getHPs();

            tklb = 0*tk0 - 30;
            tkub = 0*tk0 + 30;

            tx0 = [tm0 tk0];
            tlb = [tmlb tklb];
            tub = [tmub tkub];

            func = @(x) obj.loss(x);

            for i = 1:3
                %tx0 = tlb + (tub - tlb).*rand(1,length(tlb));

                opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'Display','off','MaxFunctionEvaluations',2000);

                [theta{i},val(i)] = fmincon(func,tx0,[],[],[],[],tlb,tub,[],opts);

                %[theta{i},val(i),~,fv] = VSGD(func,tx0,'lr',0.2,'lb',tlb,'ub',tub,'gamma',0.01,'iters',2000,'tol',1*10^(-4));

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

                obj.X = [obj.X; x];
                obj.Y = [obj.Y; y];
           
                obj = obj.condition(obj.X,obj.Y);
            end

        end

        function [E,V] = NormalQuadrature(obj,mus,sigmas)

            scale = obj.kernel.scale;

            thetas = 1./sqrt(obj.kernel.thetas{1});

            res = obj.Y - obj.mean.eval(obj.X);

            [E,V] = utils.BayesQuadNormal(scale,thetas,obj.K,obj.X,res,mus,sigmas);

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