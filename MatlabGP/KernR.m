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

classdef KernR
    
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

        function obj = KernR(kernel)
            obj.kernel = kernel;
        end

        function [y,sig] = eval(obj,x)
            
            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksf = obj.kernel.build(xs,xx);

            W = sum(ksf,2);

            y = sum(ksf.*obj.Y',2)./W;

            if nargout>1
                 if isempty(obj.mean)
                    sig=0*x(:,1);
                else

                    sig = sum(ksf.*(obj.Y'- obj.mean').^2,2)./W;

                 end

            end
        end
        
        function [y] = eval_mu(obj,x)
            
            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksf = obj.kernel.build(xs,xx);

            W = sum(ksf,2);

            y = sum(ksf.*obj.Y',2)./W;

        end

        function [sig] = eval_var(obj,x)

            if isempty(obj.mean)
                sig=0*x(:,1);
            else

                xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
                xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

                ksf = obj.kernel.build(xs,xx);

                W = sum(ksf);

                sig = sum(ksf.*(obj.Y'-obj.mean).^2,2)./W;
            end

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

            obj.kernel.scale = 1;
           
        end

        function [nll] = LL(obj,theta,XT,YT)
            
            obj.kernel = obj.kernel.setHPs(theta);

            [obj] = obj.condition(XT,YT);

            muT = obj.eval(XT);

            nll = sum((YT - muT).^2);

        end

        function [loss,dloss] = loss(obj,theta)

            tm0 = obj.mean.getHPs();
            ntm = numel(tm0);

            nV = length(theta(:));

            if nargout==2
                theta = AutoDiff(theta);
            end

            obj = obj.setHPs(theta);

            [obj] = obj.condition(obj.X,obj.Y);

            detk = det(obj.K/obj.kernel.scale + diag(0*obj.K(:,1) + obj.kernel.signn));

            loss_nll = -0.5*log(sqrt(obj.kernel.scale)) - 0.5*log(abs(detk)+eps) + 1*sum(log(eps+gampdf(abs(theta(ntm+1:end)),1.1,0.5)));

            loss_nll = -1*loss_nll;

            if nargout==2
                loss = getvalue(loss_nll);
                dloss = getderivs(loss_nll);
                dloss = reshape(full(dloss),[1 nV]);
            else
                loss = loss_nll;
            end

        end

        function [thetas,ntk,tk0] = getHPs(obj)

            tk0 = obj.kernel.getHPs();

            ntk = numel(tk0);

            thetas = tk0;

        end

        function obj = setHPs(obj,theta)

            obj.kernel = obj.kernel.setHPs(theta);

        end

        function LL = kfold(obj,theta)


            XX = obj.X;
            YY = obj.Y;

            for i = 1:3
                XT = XX;
                YT = YY;
                
                its = randsample(size(YY,1),max(floor(size(YY,1)/100),5));
                
                obj.X = XT(its,:);
                obj.Y = YT(its,:);
                
                XT(its,:) = [];
                YT(its,:) = [];
                
                LL(i) = obj.LL(theta,obj.X,obj.Y) + 2*sum(log(gampdf(theta,1.1,0.5)));
            end
            
            LL = mean(LL);
        end

        function [obj,LL] = train(obj)

            tk0 = obj.kernel.getHPs();

            tklb = 0*tk0 + 0.001;
            tkub = 0*tk0 + 10;

            tlb = tklb;
            tub = tkub;

            %func = @(x) obj.LL(x,obj.X,obj.Y);
            func = @(x) obj.kfold(x);

            [xxt,LL1] = optim.AdaptiveGridSampling(func,tlb,tub,10,20,4);

            LL = exp(1 + LL1 - max(LL1));

            theta = sum(xxt.*LL(:))/sum(LL);

            obj.kernel = obj.kernel.setHPs(theta);
            obj = obj.condition(obj.X,obj.Y,obj.lb_x,obj.ub_x);

        end

        function obj = solve(obj)

            obj.mean = obj.eval(obj.X);

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
        
        function [E,V] = NormalQuad(obj,mus,sigmas)

            scale = obj.kernel.scale;

            thetas = 1./sqrt(obj.kernel.thetas{1});

            res = obj.Y - obj.mean.eval(obj.X);

            [E,V] = utils.BayesQuadNormal(scale,thetas,obj.K,obj.X,res,mus,sigmas);

        end

        function [E,V] = BayesQuad(obj)

            theta_m = obj.mean.getHPs();
            theta_k = obj.kernel.getHPs();
            kmm = obj.kernel.qKm(obj.X,obj.lb_x,obj.ub_x,theta_k);
            km = obj.kernel.qK(obj.X,obj.lb_x,obj.ub_x,theta_k);
            kv = obj.kernel.qKq(obj.lb_x,obj.ub_x,theta_k);
            
            E = obj.mean.integrate(obj.lb_x,obj.ub_x,theta_m) + dot(kmm,obj.alpha);
            V = kv - dot(km,obj.K\km);

        end

        function sc = SquaredCorrelation(obj,x)

            [~,Vi] = obj.BayesQuad();

            sigy = obj.eval_var(x);

            theta_k = obj.kernel.getHPs();
            qKx = obj.kernel.qK(x,obj.lb_x,obj.ub_x,theta_k); 
            qKX = obj.kernel.qK(obj.X,obj.lb_x,obj.ub_x,theta_k);

            KXx = obj.kernel.build(obj.X,x);

            sc = ((qKx - dot(qKX,obj.K\KXx)).^2)./(Vi^2);;%*sigy);
        
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