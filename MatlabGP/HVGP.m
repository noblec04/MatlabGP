%{
    Heteroscedastic Variational Gaussian Process
    
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

classdef HVGP
    
    properties
        kernel
        mean
        kre

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
        E

        lb_x
        ub_x
    end

    methods

        function obj = HVGP(mean,kernel,ind)
            if isempty(mean)
                mean = means.zero;
            end
            obj.mean = mean;
            obj.kernel = kernel;
            obj.Xu = ind;

            obj.kre = VGP(mean,kernel,ind);

        end

        function [y,sig] = eval(obj,x)
            
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xu = obj.Xu;%(obj.Xu - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksu = obj.kernel.build(xs,xu);

            y = obj.mean.eval(x) + ksu*obj.alpha;

            if nargout>1
                
                v1 = obj.Kuu\ksu';
                v2 = obj.M\ksu';

                sigs = dot(ksu',v2)-dot(ksu',v1);

                [mua,siga] = obj.kre.eval(x);

                sigsa = exp(mua + siga/2);
            
                sig = obj.kernel.scale + obj.kernel.signn + sigs' + sigsa;
            end

        end

        function [y] = eval_mu(obj,x)
            
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xu = obj.Xu;%(obj.Xu - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksu = obj.kernel.build(xs,xu);

            y = obj.mean.eval(x) + ksu*obj.alpha;

        end

        function [dy] = eval_grad(obj,x)
            
            [nn,nx] = size(x);

            x = AutoDiff(x);

            y = obj.eval_mu(x);

            dy = reshape(full(getderivs(y)),[nn,nx]);

        end

        function [sig] = eval_var(obj,x)
            
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xu = obj.Xu;%(obj.Xu - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksu = obj.kernel.build(xs,xu);

            v1 = obj.Kuu\ksu';
            v2 = obj.M\ksu';

            sigs = dot(ksu',v2)-dot(ksu',v1);

            [mua,siga] = obj.kre.eval(x);

            sigsa = exp(mua + siga/2);

            sig = obj.kernel.scale + obj.kernel.signn + sigs' + sigsa;

        end

        function y = samplePrior(obj,x)
            
            K = obj.kernel.build(x,x);

            y = mvnrnd(0*x(:,1),K);

        end

        function y = samplePosterior(obj,x)
            
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xu = obj.Xu;%(obj.Xu - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksu = obj.kernel.build(xs,xu);
            kss = obj.kernel.build(xs,xs);

            sigs = -ksu*obj.Kuu\ksu' + ksu*obj.M\ksu';
            
            [mua,siga] = obj.kre.eval(x);

            sigsa = exp(mua + siga/2);

            sig = kss + obj.kernel.signn + sigs + sigsa;

            y = mvnrnd(obj.mean.eval(x)+ksu*obj.alpha,sig);
            
        end

        function obj = addInducingPoints(obj,x)
            
            replicates = ismembertol(x,obj.Xu,1e-4,'ByRows',true);

            x(replicates,:)=[];
            
            if size(x,1)>0
                obj.Xu = [obj.Xu;x];
                obj = obj.condition(obj.X,obj.Y,obj.E);
                obj.kre = obj.kre.addInducingPoints(x);
            end

        end

        function nll = nLL(obj,x,y)

            [mu,sig] = obj.eval(x);
            
            nll = sum(-log(2*pi*sqrt(abs(sig))) - ((y - mu).^2)./sig);

        end

        function [x,dy] = newXuDiff(obj)

            replicates = ismembertol((obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x),obj.Xu,1e-4,'ByRows',true);

            obj.X(replicates,:)=[];
            obj.Y(replicates)=[];
            obj.E(replicates)=[];

            Y1 = obj.eval(obj.X);

            dy = abs((obj.Y-Y1)./sqrt(obj.E));

            [~,imax] = max(dy);

            dy = dy(imax);

            x = (obj.X(imax,:) - obj.lb_x)./(obj.ub_x - obj.lb_x);

        end

        function [thetas,ntm,ntk,tm0,tk0] = getHPs(obj)

            tm0 = obj.mean.getHPs();
            tk0 = obj.kernel.getHPs();

            ntm = numel(tm0);
            ntk = numel(tk0);

            thetas = [tm0 tk0 obj.kernel.signn];

        end

        function obj = condition(obj,X,Y,E,lb,ub)

            obj.X = X;
            obj.Y = Y;
            obj.E = E;

            if nargin<5
                obj.lb_x = min(X);
                obj.ub_x = max(X);
            else
                obj.lb_x = lb;
                obj.ub_x = ub;
            end

            xf = (X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xu = obj.Xu;%(obj.Xu - obj.lb_x)./(obj.ub_x - obj.lb_x);

            obj.kernel.scale = std(Y)/2;

            obj.Kuu = obj.kernel.build(xu,xu) + (1e-6)*eye(size(xu,1));

            obj.Kuf = obj.kernel.build(xu,xf);

            obj.B = obj.Kuf*(diag(1./(obj.E + obj.kernel.signn)))*obj.Kuf';
            obj.M = obj.Kuu + obj.B;

            obj.alpha = obj.M\(obj.Kuf*((Y - obj.mean.eval(X))./(obj.E + obj.kernel.signn)));

            obj.kre = obj.kre.condition(X,log(E+eps),obj.lb_x,obj.ub_x);

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

            obj = obj.condition(obj.X,obj.Y,obj.E);

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

            if nargout==2
                theta = AutoDiff(theta);
            end

            if regress
                obj.kernel.signn = theta(ntm+ntk+1);
            end

            obj.mean = obj.mean.setHPs(theta(1:ntm));
            obj.kernel = obj.kernel.setHPs(theta(ntm+1:ntm+ntk));

            obj = obj.condition(obj.X,obj.Y);

            its = randsample(size(obj.X,1),max(5,ceil(size(obj.X,1)/50)));

            nll = obj.nLL(obj.X(its,:),obj.Y(its));
            
            nll = -1*nll;

            if nargout==2
                loss = getvalue(nll);
                dloss = getderivs(nll);
                dloss = reshape(full(dloss),[1 nV]);
            else
                loss = nll;
            end

        end

        function [obj,nll] = train(obj,regress)

            obj.kre = obj.kre.train();

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

            if regress
                obj.kernel.signn = theta(end);
                theta(end) = [];
            end

            obj.mean = obj.mean.setHPs(theta(1:ntm));
            obj.kernel = obj.kernel.setHPs(theta(ntm+1:end));
            obj = obj.condition(obj.X,obj.Y,obj.E);

        end

        function [obj,nll] = train2(obj,regress)

            obj.kre = obj.kre.train2();

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

            for i = 1:1
                %tx0 = tlb + (tub - tlb).*rand(1,length(tlb));

                [theta{i},val(i)] = VSGD(func,tx0,'lr',0.1,'lb',tlb,'ub',tub,'gamma',0.05,'iters',20,'tol',1*10^(-5));

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

        function obj = resolve(obj,x,y,e)
            
            replicates = ismembertol(x,obj.X,1e-4,'ByRows',true);

            x(replicates,:)=[];
            y(replicates)=[];
            e(replicates)=[];
            
            if size(x,1)>0

                obj.X = [obj.X; x];
                obj.Y = [obj.Y; y];
                obj.E = [obj.E; e];


                xsc = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
                xu = (obj.Xu - obj.lb_x)./(obj.ub_x - obj.lb_x);

                [k2s] = obj.kernel.build(xu,xsc);
                
                obj.B = obj.B + k2s*diag(1./(e + obj.kernel.signn))*k2s';
                obj.M = obj.Kuu + obj.B;

                obj.alpha = obj.alpha + k2s*y./(e + obj.kernel.signn);
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