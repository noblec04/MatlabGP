classdef reshape

    properties
        bias
    end

    methods

        function obj = reshape()

        end

        function [y] = forward(obj,x)

            y = x(:) + obj.bias;

        end

        function V = getHPs(obj)

            V = [obj.bias(:)];
        end

        function obj = setHPs(obj,V)

           obj.bias = reshape(V,size(obj.bias));
            
        end
    end
end