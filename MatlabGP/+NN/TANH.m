classdef TANH


    methods

        function obj = TANH()

        end

        function [y,dy] = forward(~,x)

            y = tanh(x);

            dy = sech(x).^2;

        end
    end
end