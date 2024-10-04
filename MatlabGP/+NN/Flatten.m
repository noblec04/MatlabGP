classdef Flatten

    methods

        function obj = Flatten()
            
        end

        function [y] = forward(~,x)

            y = x(:,:);

        end

    end
end