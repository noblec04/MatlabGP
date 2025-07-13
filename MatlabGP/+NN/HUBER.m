classdef HUBER

    properties
        delta
    end

    methods

        function obj = HUBER(delta)
                obj.delta = delta;
        end

        function [e,de] = forward(obj,y,yp)

            dy2 = sum(((y - yp).^2)/obj.delta);

            e = (obj.delta^2)*(sqrt(1 + dy2) - 1);

            de = -2*(y - yp);

        end
    end
end