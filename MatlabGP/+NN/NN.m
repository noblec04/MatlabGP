classdef NN

    properties
        layers
        activations
        lossfunc

        X
        Y

        lb_x
        ub_x
    end

    methods

        function obj = NN(layers,activations,loss)
            obj.layers = layers;
            obj.activations = activations;
            obj.lossfunc = loss;
        end

        function [y,obj] = forward(obj,x)

            nl = numel(obj.layers);

            y=x;

            for i = 1:nl-1
                [y] = obj.layers{i}.forward(y);
                [y] = obj.activations{i}.forward(y);
                %obj.layers{i}.out = y;
            end

            [y] = obj.layers{nl}.forward(y);

        end

        function mu = eval_mu(obj,x)
            mu = obj.predict(x);
        end

        function sig = eval_var(~,x)
            sig = 0*x(:,1);
        end

        function [mu,sig] = eval(obj,x)
            mu = obj.eval_mu(x);
            sig = obj.eval_var(x);
        end

        function [y] = predict(obj,x)

            nl = numel(obj.layers);
            
            x = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
            y=x;

            for i = 1:nl-1
                [y] = obj.layers{i}.forward(y);
                [y] = obj.activations{i}.forward(y);
            end

            [y] = obj.layers{nl}.forward(y);

        end

        function V = getHPs(obj)

            V=[];
            nl = numel(obj.layers);

            for i = 1:nl
                V1 = obj.layers{i}.getHPs();
                V = [V;V1(:)];
            end
            V = V';
        end

        function obj = setHPs(obj,V)

            nl = numel(obj.layers);

            n=1;

            for i = 1:nl

                nLs = numel(obj.layers{i}.getHPs());

                Vl = V(n:n+nLs-1);
                Vl=Vl(:);

                obj.layers{i} = obj.layers{i}.setHPs(Vl);

                n = n + nLs;
            end
            
        end

        function [e,de] = loss(obj,V,x,y)

            nV = length(V(:));

            V = AutoDiff(V(:));

            obj = obj.setHPs(V(:));

            [yp] = obj.forward(x);

            [eout] = obj.lossfunc.forward(y,yp);

            e1 = sum(eout,2);

            e = getvalue(e1);
            de = getderivs(e1);
            de = reshape(full(de),[1 nV]);

        end

        function [e,de] = Batchloss(obj,V,x,y,N)

            nV = length(V(:));

            V = AutoDiff(V(:));

            obj = obj.setHPs(V(:));

            M = 0;

            e = 0;
            de = 0;

            while size(x,1)>0
                    M=M+1;
                    itrain = randsample(size(x,1),min(N,size(x,1)));

                    xt = x(itrain,:);
                    yt = y(itrain,:);

                    x(itrain,:)=[];
                    y(itrain,:)=[];

                    [yp] = obj.forward(xt);

                    [eout] = obj.lossfunc.forward(yt,yp);

                    e1 = sum(eout,2);

                    e = e + getvalue(e1);
                    de1 = getderivs(e1);
                    de = de + reshape(full(de1),[1 nV]);
            end

            e = e/M;
            de = de/M;

        end

        function [obj,fval] = train(obj,x,y,lb,ub)%,xv,fv

            obj.X = x;
            obj.Y = y;

            if nargin<4
                obj.lb_x = min(x);
                obj.ub_x = max(x);
            else
                obj.lb_x = lb;
                obj.ub_x = ub;
            end

            x = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            tx0 = (obj.getHPs());

            func = @(V) obj.loss(V,x,y);


            opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'MaxFunctionEvaluations',3000,'MaxIterations',3000,'Display','final');
            [theta,fval] = fmincon(func,tx0,[],[],[],[],[],[],[],opts);

            %[theta,fval,xv,fv] = VSGD(func,tx0,'lr',0.01,'gamma',0.01,'iters',1000,'tol',1*10^(-7));

            obj = obj.setHPs(theta(:));
        end

        function [obj,fval] = Batchtrain(obj,x,y,N,lb,ub)%,xv,fv

            obj.X = x;
            obj.Y = y;

            if nargin<5
                obj.lb_x = min(x);
                obj.ub_x = max(x);
            else
                obj.lb_x = lb;
                obj.ub_x = ub;
            end

            x = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            tx0 = (obj.getHPs());

            func = @(V) obj.Batchloss(V,x,y,N);


            opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'MaxFunctionEvaluations',3000,'MaxIterations',3000,'Display','final');
            [theta,fval] = fmincon(func,tx0,[],[],[],[],[],[],[],opts);

            %[theta,fval,xv,fv] = VSGD(func,tx0,'lr',0.01,'gamma',0.01,'iters',1000,'tol',1*10^(-7));

            obj = obj.setHPs(theta(:));
        end
    end
end