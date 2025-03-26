classdef DynT

    properties
        weight
        biases
        scale
    end

    methods

        function obj = DynT(in,out)
            obj.weight = normrnd(zeros(out,in),sqrt(1/in));
            obj.biases = normrnd(zeros(out,1),sqrt(1/in));
            obj.scale = normrnd(zeros(1,1),sqrt(1/in));
        end

        function [y] = forward(obj,x)

            y = tanh(obj.scale*(obj.weight*x' + obj.biases)');

        end

        function V = getHPs(obj)

            V = [obj.weight(:);obj.biases(:);obj.scale(:)];

        end

        function obj = setHPs(obj,V)

            nT = numel(obj.weight(:));

            obj.weight = reshape(V(1:nT),size(obj.weight));

            obj.biases = V(nT+1:end-1);

            obj.scale = V(end);
            
        end
    end
end