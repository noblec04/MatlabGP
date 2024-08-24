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

classdef SVM
    
    properties
        kernel

        K
        alpha
        b=0;

        X
        Y

        lb_x=0;
        ub_x=1;
    end

    methods

        function obj = SVM(kernel)
            obj.kernel = kernel;
        end

        function [y] = eval(obj,x)
            
            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            at = obj.alpha.*obj.Y;

            ksf = obj.kernel.build(xs,xx);

            y = obj.b + ksf*at;

        end

        function [dy] = eval_grad(obj,x)
            
            [nn,nx] = size(x);

            x = AutoDiff(x);

            y = obj.eval(x);

            dy = reshape(full(getderivs(y)),[nn,nx]);

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

        end

        function [loss,dloss] = loss(obj,theta)

            tk0 = obj.kernel.getHPs();
            ntk = numel(tk0);

            nV = length(theta(:));

            theta = AutoDiff(theta);

            obj.kernel = obj.kernel.setHPs(theta(1:ntk)');

            nta = size(obj.X,1);

            obj.alpha = theta(ntk+1:ntk+nta);

            %obj.b = theta(end);

            [obj] = obj.condition(obj.X,obj.Y);

            s = obj.eval(obj.X);

            %err = -1*dot(s,obj.Y) + log(sum(exp(abs(obj.alpha))));

            at = obj.alpha.*obj.Y;

            err = 0.5*at'*obj.K*at - sum(obj.alpha) + dot(s,obj.Y);

            err = -1*err;

            loss = getvalue(err);
            dloss = getderivs(err);
            dloss = reshape(full(dloss),[1 nV]);

        end

        function [thetas] = getHPs(obj)

            tk0 = obj.kernel.getHPs();

            thetas = [tk0 obj.alpha];

        end

        function obj = setHPs(obj,theta)

            [tk0] = obj.kernel.getHPs();

            ntk = length(tk0);
            nta = size(obj.X,1);
            
            obj.kernel = obj.kernel.setHPs(theta(1:ntk)');
            obj.alpha = theta(ntk+1:ntk+nta);
            %obj.b = theta(end);

        end

        function [obj, LL] = train(obj)

            tk0 = obj.kernel.getHPs();

            tklb = 0*tk0 + 0.0001;
            tkub = 0*tk0 + 30;

            talb = 0*obj.Y + 0;
            taub = 0*obj.Y + 30;

            %tblb = -1*10^6;
            %tbub = 1*10^6;

            tlb = [tklb'; talb];%; tblb];
            tub = [tkub'; taub];%; tbub];

            func = @(x) obj.loss(x);

            for i = 1:3
                tx0 = tlb + (tub - tlb).*rand(length(tlb),1);

                opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'Display','off','MaxFunctionEvaluations',200,'OptimalityTolerance',1*10^(-4));

                [theta{i},val(i)] = fmincon(func,tx0,[],[],[],[],tlb,tub,[],opts);

                %[theta{i},val(i)] = VSGD(func,tx0,'lr',0.02,'lb',tlb,'ub',tub,'gamma',0.0001,'iters',2000,'tol',1*10^(-4));

            end

            [LL,i] = min(val);

            theta = theta{i};

            obj = obj.setHPs(theta);
            obj = obj.condition(obj.X,obj.Y);

            w = obj.eval(obj.X);

            obj.b = -1*mean(w);

        end
    end
end