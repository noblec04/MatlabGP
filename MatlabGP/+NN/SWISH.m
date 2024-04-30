classdef SWISH

    properties
        beta
    end

    methods

        function obj = SWISH(beta)
            obj.beta = beta;
        end

        function [y,dy] = forward(obj,x)

            y = x./(1 + exp(-obj.beta*x));

            dy = (exp(obj.beta*x).*(obj.beta*x +exp(obj.beta*x)+1))./(1 + exp(obj.beta*x)).^2;

        end
    end
end