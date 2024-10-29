classdef RRNN 
    %{
        Residual Random Neural Networks
        M. Andrecut
        October 24, 2023
    %}
    
    properties
        Win;
        activation;
        hidden;
        Wout;

        lb_x
        ub_x
        dim

        lb_y
        ub_y
    end
    
    methods
        function obj = RRNN(act,hidden,lb,ub)
            
            obj.activation = act;
            obj.hidden = hidden;
            obj.lb_x = lb;
            obj.ub_x = ub;

            obj.dim = length(lb);

        end

        function [obj,Ri] = train(obj,X,y,T)
            
            X = obj.scale(X);
            
            obj.lb_y = min(y,[],1);
            obj.ub_y = max(y,[],1);

            y = obj.scale_y(y);

            if nargin<4
                T=5;
            end

            R = y;

            for i = 1:T
                
                obj.Win{i} = rand(size(X,2)+1, obj.hidden) * 2 - 1;
                H{i} = obj.activation.forward([X, ones(size(X,1),1)] * obj.Win{i});
                
                obj.Wout{i} = lsqminnorm([H{i}, ones(size(H{i},1),1)],R);

                Ri(i,:) = sum(abs(R),1);

                R = R - [H{i}, ones(size(H{i},1),1)]*obj.Wout{i};

            end

        
        end

        function x = scale(obj,x)
            
            x = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

        end

        function y = scale_y(obj,y)
            
            y = (y - obj.lb_y)./(obj.ub_y - obj.lb_y);

        end

        function y = unscale_y(obj,y)
            
            y = obj.lb_y + (obj.ub_y - obj.lb_y).*y;

        end
        
        function y = eval(obj, X)
            %PREDICT Predicts the output of the trained model for new input
            %data
            % Inputs:
            %   obj - trained RRNN
            %   X - Input data
            
            % Output:
            %   y - output output
            
            y = 0;
            X = obj.scale(X);

            for i = 1:numel(obj.Win)

                H = obj.activation.forward([X, ones(size(X,1),1)] * obj.Win{i});
                y = y + [H, ones(size(H,1),1)] * obj.Wout{i};

            end

            y = obj.unscale_y(y);
        end

        function sig = eval_var(obj,X)
            %PREDICT Predicts the output of the trained model for new input
            %data
            % Inputs:
            %   obj - trained RRNN
            %   X - Input data
            
            % Output:
            %   y - output output
            
            H = obj.activation.forward([X, ones(size(X,1),1)] * obj.Win{1});
            y1 = [H, ones(size(H,1),1)] * obj.Wout{1};

            y = 0;
            X = obj.scale(X);

            for i = 1:numel(obj.Win)

                H = obj.activation.forward([X, ones(size(X,1),1)] * obj.Win{i});
                y = y + [H, ones(size(H,1),1)] * obj.Wout{i};

            end

            y = obj.unscale_y(y);
            y1 = obj.unscale_y(y1);

            sig = (y1 - y).^2;
        end

        function [mu,sig] = eval_all(obj,x)
            mu = obj.eval(x);
            sig = obj.eval_var(x);
        end
    end
end
