classdef SNAKE

    properties
        beta
    end

    methods

        function obj = SNAKE(beta)
            obj.beta = beta;
        end

        function [y,dy] = forward(obj,x)

            y = x+sin(obj.beta*x)/obj.beta;

            dy = 1 + cos(obj.beta*x);

        end
    end
end