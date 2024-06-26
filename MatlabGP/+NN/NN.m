classdef NN

    properties
        layers
        activations
        lossfunc

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
                obj.layers{i}.out = y;
            end

            [y] = obj.layers{nl}.forward(y);

        end

        function [y,obj] = predict(obj,x)

            nl = numel(obj.layers);
            
            x = (x' - obj.lb_x)./(obj.ub_x - obj.lb_x);
            y=x';

            for i = 1:nl-1
                [y] = obj.layers{i}.forward(y);
                [y] = obj.activations{i}.forward(y);
                obj.layers{i}.out = y;
            end

            [y] = obj.layers{nl}.forward(y);

        end

        function [obj] = backward(obj,de)

            nl = numel(obj.layers);

            obj.layers{nl}.sensitivity = de;

            for i = nl-1:-1:1
                [~,da] = obj.activations{i}.forward(obj.layers{i}.out);
                obj.layers{i}.sensitivity = diag(da)*obj.layers{i+1}.weight'*obj.layers{i+1}.sensitivity;
            end
            
        end

        function dy = getGrads(obj,x)
                nl = numel(obj.layers);
                
                dy = [];

                for i = 1:nl
                    db = obj.layers{i}.sensitivity;
                    if i==1
                        a = x;
                    else
                        a = obj.layers{i-1}.out;
                    end
                    dw = obj.layers{i}.sensitivity*a';
                    dy = [dy;dw(:);db(:)];
                end
        end

        function V = getHPs(obj)

            V=[];
            nl = numel(obj.layers);

            for i = 1:nl
                V1 = obj.layers{i}.getHPs();
                V = [V;V1(:)];
            end
        end

        function obj = setHPs(obj,V)

            nl = numel(obj.layers);

            for i = 1:nl
                nLs = numel(obj.layers{i}.getHPs());

                Vl = V(1:nLs);

                obj.layers{i} = obj.layers{i}.setHPs(Vl);

                V(1:nLs)=[];
            end
            
        end

        function [e,de] = loss(obj,V,x,y)

            obj = obj.setHPs(V(:));

            nx = size(x,1);

            for i = 1:nx

                xa = x(i,:);

                [yp(i,:),obj] = obj.forward(xa');

                [e(i,:),de] = obj.lossfunc.forward(y(i,:),yp(i,:));

                [obj] = obj.backward(de);

                dy(i,:) = obj.getGrads(xa');
            end

            e = sum(e,"all");
            de = sum(dy,1);

        end

        function [obj,fval,xv,fv] = train(obj,x,y)

            obj.lb_x = min(x);
            obj.ub_x = max(x);

            x = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            tx0 = (obj.getHPs())';

            func = @(V) obj.loss(V,x,y);

            [theta,fval,xv,fv] = VSGD(func,tx0,'lr',0.01,'gamma',0.1,'iters',3000,'tol',1*10^(-7));

            obj = obj.setHPs(theta(:));
        end
    end
end