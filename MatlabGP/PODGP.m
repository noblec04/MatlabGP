%{
    POD - Gaussian Process
%}

classdef PODGP
    
    properties
        GP
        POD

        x
        Y

        lb_x
        ub_x
    end

    methods

        function obj = PODGP(GP,POD)
            
            obj.GP = GP;
            obj.POD = POD;
            
        end

        function [y] = eval(obj,x)
            
            coeffs = obj.GP.eval(x);
            y = obj.POD.eval(coeffs);

        end

        function [y] = eval_var(obj,x)
            
            coeffs = obj.GP.eval_var(x);
            y = obj.POD.eval_var(coeffs);

        end

        function [dy,dsig] = eval_grad(obj,x)
            
            [nn,nx] = size(x);

            A = size(obj.POD.Y);

            A(1) = nn;

            x = AutoDiff(x);

            y = obj.eval(x);

            dy = squeeze(reshape(full(getderivs(y)),[A nx]));

            if nargout==2
                sig = obj.eval_var(x);

                dsig = squeeze(reshape(full(getderivs(sig)),[A nx]));
            end

        end

        function [thetas] = getHPs(obj)

            thetas = obj.GP.getHPs();

        end

        function obj = setHPs(obj,thetas)

            obj.GP = obj.GP.setHPs(thetas);

        end

        function [obj,L] = train(obj,x,Y)

            obj.lb_x = min(x);
            obj.ub_x = max(x);

            obj.POD = obj.POD.train(Y);

            y = obj.POD.score(:,1:obj.POD.nmodes);

            obj.GP = obj.GP.condition(x,y,obj.lb_x,obj.ub_x);
            [obj.GP,L] = obj.GP.train();

        end

    end
end