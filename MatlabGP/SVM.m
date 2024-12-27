%{
    Support Vector Machine.

    A kernel function can be created from which the SVM can be
    generated.

    The SVM can then be conditioned on training data.

%}

classdef SVM
    
    properties
        kernel

        K
        alpha
        alphas
        b=0;

        SV
        SVs

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
            
            xx = (obj.SVs - obj.lb_x)./(obj.ub_x - obj.lb_x);
            xs = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            ksf = obj.kernel.build(xs,xx);

            y = obj.b + ksf*obj.alphas;

        end

        function [dy] = eval_grad(obj,x)
            
            [nn,nx] = size(x);

            x = AutoDiff(x);

            y = obj.eval(x);

            dy = reshape(full(getderivs(y)),[nn,nx]);

        end


        function [obj] = condition(obj,X,Y,lb,ub)

            Y = 2*((Y>0)-0.5);

            obj.X = X;
            obj.Y = Y;

            if nargin<4
                obj.lb_x = min(X);
                obj.ub_x = max(X);
            else
                obj.lb_x = lb;
                obj.ub_x = ub;
            end

        end

        function [obj] = train(obj)
            

            xx = (obj.X - obj.lb_x)./(obj.ub_x - obj.lb_x);

            obj.kernel.scale = 1;

            [obj.K] = obj.kernel.build(xx,xx);

            H = (obj.Y*obj.Y').*obj.K;

            opt=optimset('algorithm','interior-point-convex','TolFun',1e-6,'TolX',1e-6,'TolCon',1e-6,'display','off');
            
            F=-ones(size(obj.X,1),1);
            obj.alpha=quadprog(H,F,[],[],obj.Y',0,zeros(size(obj.X,1),1),(1e4)*ones(size(obj.X,1),1),[],opt);
            
            obj.alpha(obj.alpha<1e-4)=0;
            obj.b=mean(obj.Y(obj.alpha>0,:)-obj.kernel.build(obj.X(obj.alpha>0,:),obj.X(obj.alpha>0,:))*(obj.Y(obj.alpha>0,:).*obj.alpha(obj.alpha>0,:)));

            obj.SV=obj.alpha>0;
            obj.SVs=obj.X(obj.SV,:);
            obj.alphas=obj.Y(obj.SV,:).*obj.alpha(obj.SV,:);

        end

    end
end