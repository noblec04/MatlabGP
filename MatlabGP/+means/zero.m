classdef zero<means.means

    methods
        function obj = zero()
            obj.coeffs{1} = 0;
            obj.meanz{1} = obj;
        end

        function [y,dy] = forward(~,x)
            y = 0*x(:,1);
            dy = 0*x;
        end

    end
end