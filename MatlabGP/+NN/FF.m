classdef FF

    properties
        weight
        biases
        sensitivity
        out
    end

    methods

        function obj = FF(in,out)
            obj.weight = normrnd(zeros(out,in),sqrt(1/in));
            obj.biases = normrnd(zeros(out,1),sqrt(1/in));
        end

        function [y] = forward(obj,x)

            y = (obj.weight*x' + obj.biases)';

        end

        function V = getHPs(obj)

            V = [obj.weight(:);obj.biases(:)];
        end

        function obj = setHPs(obj,V)

            nT = numel(obj.weight(:));

            obj.weight = reshape(V(1:nT),size(obj.weight));

            obj.biases = V(nT+1:end);
            
        end
    end
end