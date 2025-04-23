classdef NN

    properties
        layers
        activations
        lossfunc

        X
        Y

        lb_x
        ub_x

        lb_y = 0;
        ub_y = 1;
    end

    methods

        function obj = NN(layers,activations,loss)
            obj.layers = layers;
            obj.activations = activations;
            obj.lossfunc = loss;
        end

        function y = set_eval(obj,V,x)

            V = AutoDiff(V(:));

            obj = obj.setHPs(V(:));

            [y] = obj.forward(x);

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

        function [mu,sig] = MCeval(obj,x,V)

            [nMC,~] = size(V);

            for i = 1:nMC

                obj2 = obj.setHPs(V(:,i));

                y(i,:,:) = obj2.eval_mu(x);

            end

            mu = squeeze(mean(y,1));
            sig = squeeze(std(y,[],1));

        end

        function [y] = predict(obj,x)

            x = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            nl = numel(obj.layers);

            y=x;

            for i = 1:nl-1
                [y] = obj.layers{i}.forward(y);
                [y] = obj.activations{i}.forward(y);
            end

            [y] = obj.layers{nl}.forward(y);

            y = obj.lb_y + (obj.ub_y - obj.lb_y).*y;

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

        function [y,dy] = valueAndGrad(obj,V,x)

            nV = length(V(:));

            V = AutoDiff(V(:));

            obj = obj.setHPs(V(:));

            x = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            [yp] = obj.forward(x);

            y = obj.lb_y + (obj.ub_y - obj.lb_y).*y;

            y = getvalue(yp);

            ny = numel(y);

            dy = getderivs(yp);
            dy = reshape(full(dy),[ny nV]);

        end

        function [e,de] = loss(obj,V,x,y)

            nV = length(V(:));

            if nargout == 2
                V = AutoDiff(V(:));
            end

            obj = obj.setHPs(V(:));

            [yp] = obj.forward(x);

            [eout] = obj.lossfunc.forward(y,yp);

            e1 = sum(eout,2);

            if nargout == 2
                e = getvalue(e1);
                de = getderivs(e1);
                de = reshape(full(de),[1 nV]);
            else
                e = e1;
            end

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

            obj.lb_y = min(y);
            obj.ub_y = max(y);

            x = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
            y = (y - obj.lb_y)./(obj.ub_y - obj.lb_y);

            tx0 = (obj.getHPs());

            func = @(V) obj.loss(V,x,y);


            opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'MaxFunctionEvaluations',500,'MaxIterations',500,'Display','off');
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

            obj.lb_y = min(y);
            obj.ub_y = max(y);

            x = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
            y = (y - obj.lb_y)./(obj.ub_y - obj.lb_y);

            tx0 = (obj.getHPs());

            func = @(V) obj.Batchloss(V,x,y,N);


            opts = optimoptions('fmincon','SpecifyObjectiveGradient',true,'MaxFunctionEvaluations',3000,'MaxIterations',3000,'Display','final');
            [theta,fval] = fmincon(func,tx0,[],[],[],[],[],[],[],opts);

            %[theta,fval,xv,fv] = VSGD(func,tx0,'lr',0.01,'gamma',0.01,'iters',1000,'tol',1*10^(-7));

            obj = obj.setHPs(theta(:));
        end

        function draw(obj)

            nL = numel(obj.layers);

            figure

            minw = 0;
            maxw = 0;

            for i = 1:nL
                
                minw = min(minw,min(obj.layers{i}.weight(:)));
                maxw = max(maxw,max(obj.layers{i}.weight(:)));

            end

            maxw = max(abs(minw),maxw);

            n = 0;
            for i = 1:nL
                n=n+1;
                subplot(1,2*nL,n)
                imagesc(obj.layers{i}.weight)
                colorbar
                utils.cmocean('balance')
                axis tight
                axis equal
                axis off
                clim([-1*maxw maxw])

                n = n+1;
                subplot(1,2*nL,n)
                plot(obj.layers{i}.biases,1:length(obj.layers{i}.biases))

            end

        end

        function plotThroughput(obj,x)

            nL = numel(obj.layers);

            x = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            x0 = x;
            %figure

            for i = 1:nL-1

                subplot(1,nL,i)
                hold on
                x = obj.layers{i}.forward(x);
                x = obj.activations{i}.forward(x);
                waterfall(x0',[1:size(x,2)]',x')
                view(20,60)
                axis square
                grid on
                box on
            end

            subplot(1,nL,nL)
            hold on
            x = obj.layers{nL}.forward(x);
            plot(x0,x)
            axis square
            grid on
            box on
        end

    end
end