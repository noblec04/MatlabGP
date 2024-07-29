%{
    Variational Gaussian Process
    
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

classdef VGP
    
    properties
        kernel
        mean

        B
        
        M
        Minv

        Kuf

        Kuu
        Kuuinv

        alpha
        
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

        function [sig] = eval_var(obj,x)
            
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xu = (obj.Xu - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksu = obj.kernel.build(xs,xu);

            sigs = -dot(ksu',(obj.Kuuinv)*ksu') + dot(ksu',obj.Minv*ksu');

            sig = obj.kernel.scale + obj.kernel.signn + sigs';

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

        function obj = addInducingPoints(obj,x)
            
            replicates = ismembertol(x,obj.Xu,1e-4,'ByRows',true);

            x(replicates,:)=[];
            
            if size(x,1)>0
                obj.Xu = [obj.Xu;x];
                obj = obj.condition(obj.X,obj.Y);
            end

        end

        function nll = nLL(obj,x,y)

            [mu,sig] = obj.eval(x);
            
            nll = sum(-log(2*pi*sqrt(abs(sig))) - ((y - mu).^2)./sig);

        end

        function dy = newXuDiff(obj,x)

            obj2 = obj.addInducingPoints(x);

            Y1 = obj.eval(obj.X);
            Y2 = obj2.eval(obj.X);

            dy = -1*abs(sum(Y2-Y1));

        end

        function [thetas,ntm,ntk,tm0,tk0] = getHPs(obj)

            tm0 = obj.mean.getHPs();
            tk0 = obj.kernel.getHPs();

            ntm = numel(tm0);
            ntk = numel(tk0);

            thetas = [tm0 tk0 obj.kernel.signn];

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
            obj.Kuuinv = pinv(obj.Kuu,1*10^(-7));

            obj.Kuf = obj.kernel.build(xu,xf);

            obj.B = obj.Kuf*obj.Kuf'/obj.kernel.signn;
            obj.M = obj.Kuu + obj.B;
            obj.Minv = pinv(obj.M,1*10^(-7));

            obj.alpha = obj.Minv*obj.Kuf*(Y - obj.mean.eval(X))/obj.kernel.signn;

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

            obj = obj.condition(obj.X,obj.Y);

            its = randsample(size(obj.X,1),max(5,ceil(size(obj.X,1)/50)));

            nll = obj.nLL(obj.X(its,:),obj.Y(its));
            
            nll = -1*nll;

            nll(isnan(nll)) = 0;
            nll(isinf(nll)) = 0;
        end

        function [loss, dloss] = loss(obj,theta,regress)

            nV = length(theta(:));
            tm0 = obj.mean.getHPs();
            ntm = numel(tm0);
            tk0 = obj.kernel.getHPs();
            ntk = numel(tk0);

            theta = AutoDiff(theta);

            if regress
                obj.kernel.signn = theta(ntm+ntk+1);
            end

            obj.mean = obj.mean.setHPs(theta(1:ntm));
            obj.kernel = obj.kernel.setHPs(theta(ntm+1:ntm+ntk));

            obj = obj.condition(obj.X,obj.Y);

            its = randsample(size(obj.X,1),max(5,ceil(size(obj.X,1)/50)));

            nll = obj.nLL(obj.X(its,:),obj.Y(its));
            
            nll = -1*nll;

            loss = getvalue(nll);
            dloss = getderivs(nll);
            dloss = reshape(full(dloss),[1 nV]);

        end

        function [obj,nll] = train(obj,regress)

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


            xxt = tlb + (tub - tlb).*lhsdesign(500*length(tlb),length(tlb));

            for ii = 1:size(xxt,1)
                LL(ii) = -1*func(xxt(ii,:));
            end

            LL = exp(1 + LL - max(LL));

            theta = sum(xxt.*LL')/sum(LL);

            nll = sum(LL);

            % opts = bads('Defaults');
            % opts.Display = 'final';
            % opts.TolFun = 10^(-2);
            % opts.TolMesh = 10^(-2);
            % 
            % for i = 1:3
            %     tx0 = tlb + (tub - tlb).*rand(1,length(tlb));
            % 
            %     [theta{i},val(i)] = bads(func,tx0,tlb,tub,tlb,tub,[],opts);
            % 
            % end
            % 
            % [nll,i] = min(val);
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