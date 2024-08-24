classdef SWISH

    properties
        beta
    end

    methods

        function obj = SWISH(beta)
            obj.beta = beta;
        end

        function [y] = forward(obj,x)

            y = x./(1 + exp(-obj.beta*x));

        end
    end
end