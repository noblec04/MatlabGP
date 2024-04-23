classdef const<means.means

    methods
        function obj = const(coeffs)
            obj.coeffs{1} = coeffs;
            obj.meanz{1} = obj;
        end

        function [y,dy] = forward(~,x,theta)
            y = theta + 0*x(:,1);
            dy = 0*x;
        end

    end
end