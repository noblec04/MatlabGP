classdef SNAKE

    properties
        beta
    end

    methods

        function obj = SNAKE(beta)
            obj.beta = beta;
        end

        function [y] = forward(obj,x)

            y = x+sin(obj.beta*x)/obj.beta;

        end
    end
end