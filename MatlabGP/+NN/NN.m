classdef NN

    properties
        layers
        activations
    end

    methods

        function obj = NN(layers,activations)
            obj.layers = layers;
            obj.activations = activations;
        end

        function [y,obj] = forward(obj,x)

            nl = numel(obj.layers);

            y = obj.layers{1}.forward(x);
            [y] = obj.activations{1}.forward(y);
            obj.layers{1}.out = y;

            for i = 2:nl-1
                [y] = obj.layers{i}.forward(y);
                [y] = obj.activations{i}.forward(y);
                obj.layers{i}.out = y;
            end

            [y] = obj.layers{nl}.forward(y);

        end

        function [obj] = backward(obj,e)

            nl = numel(obj.layers);

            obj.layers{nl}.sensitivity = -2*e;

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

            obj = obj.setHPs(V);

            its = 1:length(x);%randsample(size(x,1),max(5,ceil(size(x,1)/50)));

            nx = length(its);

            for i = 1:nx

                j = its(i);

                [yp(i,:),obj] = obj.forward(x(j,:));

                e(i,:) = y(j,:) - yp(i,:);

                [obj] = obj.backward(e(i,:));

                dy(i,:) = obj.getGrads(x(j,:));
            end

            e = sum((y(its,:) - yp).^2,"all");
            de = sum(dy,1)';

        end

        function [obj,fval,xv,fv] = train(obj,x,y)

            tx0 = obj.getHPs();

            func = @(V) obj.loss(V,x,y);

            [theta,fval,xv,fv] = VSGD(func,tx0,'lr',0.01,'gamma',0.01,'iters',5000,'tol',1*10^(-7));

            obj = obj.setHPs(theta);
        end
    end
end