classdef RELU


    methods

        function obj = RELU()

        end

        function [y,dy] = forward(~,x)

            y = 0.5*(x + abs(x));

            dy = (sign(x)+1)/2;

        end
    end
end