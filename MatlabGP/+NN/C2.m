classdef C2

    properties
        filter
        bias
    end

    methods

        function obj = C2(F)
            obj.filter = normrnd(zeros(F,F),sqrt(1/F));
        end

        function [y] = forward(obj,x)

            y = utils.conv2D(x,obj.filter,obj.bias);

        end

        function V = getHPs(obj)

            V = [obj.filter(:);obj.bias(:)];
        end

        function obj = setHPs(obj,V)

            nT = numel(obj.filter(:));

            obj.filter = reshape(V(1:nT),size(obj.filter));

            obj.bias = reshape(V(nT+1:end),size(obj.bias));
            
        end
    end
end