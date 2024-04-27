classdef sine<means.means

    properties
        dims
        periods
        phases
    end

    methods
        function obj = sine(dims,periods,phases,amps)
            obj.dims = dims;
            obj.periods = periods;
            obj.phases = phases;
            obj.coeffs{1} = amps;
            obj.meanz{1} = obj;
        end

        function [y,dy] = forward(obj,x,theta)
            y = sum(theta.*sin(obj.periods.*x(:,obj.dims) + obj.phases),2);
            dy = sin(obj.periods.*x(:,obj.dims) + obj.phases);
        end

    end
end