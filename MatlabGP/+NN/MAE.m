classdef MAE

    properties

    end

    methods

        function obj = MAE()

        end

        function [e] = forward(~,y,yp)

            e = log(sum(exp(abs(y - yp))));

        end
    end
end