classdef linear<means.means

    methods
        function obj = linear(coeffs)
            obj.coeffs{1} = coeffs;
            obj.meanz{1} = obj;
        end

        function [y,dy] = forward(~,x,theta)
            y = (theta*x')';
            dy = x;
        end

    end
end