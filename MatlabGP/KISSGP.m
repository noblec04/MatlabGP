%{
    KISS Gaussian Process
    
    An approximate Gaussian process model that uses inducing points to
    create a low rank approximate conditional predictive distribution.

    This class allows the user to pass in a mean and kernel model and a set
    of inducing points.

    The model can then be conditioned on training data, x and y.

    The model can then be trained to optimize hyperparameters of the mean
    and the kernel.

    The class provides a method for adding inducing points from the
    training set that maximize the change in the predictive distribution.

%}

classdef KISSGP
    
    properties
        kernel
        mean

        training = 0;
        
        M

        K
        R
        vg

        alpha
        
        X
        Xu
        Xg
        Y

        lb_x
        ub_x

        lb_y
        ub_y
    end

    methods

        function obj = KISSGP(mean,kernel,nind)
            if isempty(mean)
                mean = means.zero;
            end
            obj.mean = mean;
            obj.kernel = kernel;
            obj.Xu = nind;
        end

        function [thetas,ntm,ntk,tm0,tk0] = getHPs(obj)

            tm0 = obj.mean.getHPs();
            tk0 = obj.kernel.getHPs();

            ntm = numel(tm0);
            ntk = numel(tk0);

            thetas = [tm0 tk0 obj.kernel.signn];

        end

        function obj = BuildKern(obj)
            
            [~,nd] = size(obj.X);

            for i = 1:nd
                obj.Xg{i} = linspace(0,1,obj.Xu)';
                obj.K{i} = obj.kernel.build(obj.Xg{i},obj.Xg{i});
            end
        end

        function obj = condition(obj,X,Y,lb,ub)

            obj.X = X;
            obj.Y = Y;

            if nargin<4
                obj.lb_x = min(X);
                obj.ub_x = max(X);
            else
                obj.lb_x = lb;
                obj.ub_x = ub;
            end

            obj.lb_y = min(Y);
            obj.ub_y = max(Y);

            xf = (X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            yf = (Y - obj.lb_y)./(obj.ub_y - obj.lb_y);
            
            obj.kernel.scale = std(yf);

            obj = obj.BuildKern();
            obj.M = utils.interpgrid(obj.Xg,xf,3);

            ff = @(x) obj.M*utils.kronmvm(obj.K,obj.M'*x) + obj.kernel.signn*x;

            res = yf - obj.mean.eval(obj.X);

            [obj.alpha,flag] = utils.conjgrad(ff,res,1*10^(-5),100);

            if ~obj.training
                obj = obj.LOVE();
            end

        end

        function obj = LOVE(obj)
            N = size(obj.M,2);
            v = obj.M*utils.kronmvm(obj.K,ones(N,1)/N ); 
            n = size(obj.M,1); 
            a = 1./obj.kernel.signn.^2;
            [Q,T] = utils.lanczos_arpack(@(x)obj.M*utils.kronmvm(obj.K,obj.M'*x)+a.*x, v, min(50,n-1));

            obj.R = utils.kronmvm(obj.K,obj.M'*Q)/chol(T); 
            
            obj.vg = sum(obj.R.^2,2);

        end

        function [mu,sig] = eval(obj,xs)

            Ms = utils.interpgrid(obj.Xg,xs,3);

            mu = obj.mean.eval(xs) + obj.lb_y + (obj.ub_y - obj.lb_y)*Ms*utils.kronmvm(obj.K,obj.M'*obj.alpha);
            
            if nargout>1
                if~isempty(obj.vg)
                    ve = Ms*obj.vg;
                    sig = obj.kernel.signn + max(obj.kernel.scale - ve,0);
                else
                    sig = obj.kernel.signn + 0*xs(:,1);
                end
            end
        end

        function [mu] = eval_mu(obj,xs)

            Ms = utils.interpgrid(obj.Xg,[xs;xs],3);

            mu = obj.mean.eval([xs;xs]) + obj.lb_y + (obj.ub_y - obj.lb_y)*Ms*utils.kronmvm(obj.K,obj.M'*obj.alpha);

            mu = mu(1:ceil(end/2));%
        end

        function [sig] = eval_var(obj,xs)

            Ms = utils.interpgrid(obj.Xg,[xs;xs],3);

            if~isempty(obj.vg)
                ve = Ms*obj.vg;
                sig = obj.kernel.signn + max(obj.kernel.scale - ve,0);
                sig = sig(1:ceil(end/2));
            else
                sig = 0*xs(:,1);
            end
        end

        function nll = LL(obj,theta,regress)

            if regress
                obj.kernel.signn = theta(end);
            end

            tm0 = obj.mean.getHPs();
            ntm = numel(tm0);
            tk0 = obj.kernel.getHPs();
            ntk = numel(tk0);
            
            obj.mean = obj.mean.setHPs(theta(1:ntm));
            obj.kernel = obj.kernel.setHPs(theta(ntm+1:ntm+ntk));

            its = randsample(size(obj.X,1),min(size(obj.X,1),100));

            obj = obj.condition(obj.X(its,:),obj.Y(its));

            nll = obj.nLL(obj.X,obj.Y);
            
            nll = -1*nll;

            nll(isnan(nll)) = 0;
            nll(isinf(nll)) = 0;
        end

        function nll = nLL(obj,x,y)

            [mu] = obj.eval(x);
            
            nll = sum(((y - mu).^2));%./sig);  -log(2*pi*sqrt(abs(sig))) +

        end

        function [obj,nll] = train(obj,regress)

            obj.training = 1;

            if obj.kernel.signn==0||nargin<2
                regress=1;
            end
           
            tm0 = obj.mean.getHPs();
            ntm = numel(tm0);

            tmlb = 0*tm0 - 10;
            tmub = 0*tm0 + 10;

            tk0 = obj.kernel.getHPs();

            tklb = 0*tk0 + 0.001;
            tkub = 0*tk0 + 2;

            tlb = [tmlb tklb];
            tub = [tmub tkub];

            if regress
                tlb(end+1) = 0.001;
                tub(end+1) = std(obj.Y)/5;
            end

            func = @(x) obj.LL(x,regress);


            xxt = tlb + (tub - tlb).*lhsdesign(200*length(tlb),length(tlb));

            for ii = 1:size(xxt,1)
                LL(ii) = func(xxt(ii,:));
            end

            LL = exp(1 + LL - max(LL));

            theta = sum(xxt.*LL')/sum(LL);

            % [~,ii] = max(LL);
            % 
            % theta = xxt(ii,:);

            nll = sum(LL);

            if regress
                obj.kernel.signn = theta(end);
                theta(end) = [];
            end

            obj.training = 0;

            obj.mean = obj.mean.setHPs(theta(1:ntm));
            obj.kernel = obj.kernel.setHPs(theta(ntm+1:end));
            obj = obj.condition(obj.X,obj.Y);

        end

        function [obj,nll] = train2(obj,regress)

            if obj.kernel.signn==0||nargin<2
                regress=1;
            end
           
            tm0 = obj.mean.getHPs();
            ntm = numel(tm0);

            tmlb = 0*tm0 - 10;
            tmub = 0*tm0 + 10;

            tk0 = obj.kernel.getHPs();

            tklb = 0*tk0 + 0.001;
            tkub = 0*tk0 + 2;

            tlb = [tmlb tklb];
            tub = [tmub tkub];

            tx0 = [tm0 tk0];

            if regress
                tlb(end+1) = 0.001;
                tub(end+1) = std(obj.Y)/5;
                tx0=[tx0 0.1];
            end

            func = @(x) obj.loss(x,regress);

            for i = 1:3
                %tx0 = tlb + (tub - tlb).*rand(1,length(tlb));

                [theta{i},val(i)] = VSGD(func,tx0,'lr',0.01,'lb',tlb,'ub',tub,'gamma',0.5,'iters',200,'tol',1*10^(-5));

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

        function obj = resolve(obj,x,y)
            
            replicates = ismembertol(x,obj.X,1e-4,'ByRows',true);

            x(replicates,:)=[];
            y(replicates)=[];
            
            if size(x,1)>0

                obj.X = [obj.X; x];
                obj.Y = [obj.Y; y];

                xsc = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
                xu = (obj.Xu - obj.lb_x)./(obj.ub_x - obj.lb_x);

                [k2s] = obj.kernel.build(xu,xsc);
                
                obj.B = obj.B + k2s*k2s'/obj.kernel.signn;
                obj.M = obj.Kuu + obj.B;
                obj.Minv = pinv(obj.M);

                obj.alpha = obj.alpha + k2s*y/obj.kernel.signn;
            end
            
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