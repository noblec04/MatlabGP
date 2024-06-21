classdef LIN

    properties
    end

    methods

        function obj = LIN()
        end

        function [y,dy] = forward(~,x)

            y = x;

            dy = 1;

        end
    end
end