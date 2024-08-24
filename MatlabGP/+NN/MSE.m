classdef MSE

    properties

    end

    methods

        function obj = MSE()

        end

        function [e,de] = forward(~,y,yp)

            e = sum((y - yp).^2);

            de = -2*(y - yp);

        end
    end
end