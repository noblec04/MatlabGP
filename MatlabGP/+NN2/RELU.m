classdef RELU


    methods

        function obj = RELU()

        end

        function [y] = forward(~,x)

            y = 0.5*(x + abs(x));

        end
    end
end