classdef C1

    properties
        filter
        bias=0;
    end

    methods

        function obj = C1(F)
            obj.filter = normrnd(zeros(F,1),sqrt(1/F));
        end

        function [y] = forward(obj,x)

            y = utils.conv1D(x,obj.filter,obj.bias);

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