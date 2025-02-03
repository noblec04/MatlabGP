%{
    Regression Neural Network
%}

classdef PODNN
    
    properties
        NN
        POD

        x
        Y

        lb_x
        ub_x

    end

    methods

        function obj = PODNN(NN,POD)
            
            obj.NN = NN;
            obj.POD = POD;


        end

        function [y] = eval(obj,x)
            
            x = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);
            coeffs = obj.NN.eval(x);
            y = obj.POD.eval(coeffs);

        end

        function [dy] = eval_grad(obj,x)
            
            [nn,nx] = size(x);

            x = AutoDiff(x);

            y = obj.eval(x);

            dy = reshape(full(getderivs(y)),[nn,nx]);

        end

        function [thetas] = getHPs(obj)

            thetas = obj.NN.getHPs();

        end

        function obj = setHPs(obj,thetas)

            obj.NN = obj.NN.setHPs(thetas);

        end

        function [obj,L] = train(obj,x,Y)

            obj.lb_x = min(x);
            obj.ub_x = max(x);

            x = (x - obj.lb_x)./(obj.ub_x - obj.lb_x);

            obj.POD = obj.POD.train(Y);

            y = obj.POD.score(:,1:obj.POD.nmodes);
            [obj.NN,L] = obj.NN.train(x,y,obj.lb_x,obj.ub_x);

        end

    end
end