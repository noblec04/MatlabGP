classdef const<means.means

    methods
        function obj = const(coeffs)
            obj.coeffs{1} = coeffs;
            obj.meanz{1} = obj;
        end

        function [y,dy] = forward(~,x,theta)
            y = theta + 0*x(:,1);
            dy = 0*x(:,1);
        end

        function I = integrate(~,lb,ub,theta)

            I = theta*prod(ub - lb); 

        end

    end
end