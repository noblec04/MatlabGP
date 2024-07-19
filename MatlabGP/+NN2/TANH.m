classdef TANH


    methods

        function obj = TANH()

        end

        function [y] = forward(~,x)

            y = tanh(x);

        end
    end
end