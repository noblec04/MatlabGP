classdef FAN

    properties
        weight1
        weight2
        biases
        sensitivity
        out
    end

    methods

        function obj = FAN(in,out,modes)
            obj.weight1 = normrnd(zeros(modes,in),sqrt(1/in));
            obj.weight2 = normrnd(zeros(out-2*modes,in),sqrt(1/in));
            obj.biases = normrnd(zeros(out-2*modes,1),sqrt(1/in));
        end

        function [y] = forward(obj,x)

            y1 = cos(obj.weight1*x')';
            y2 = sin(obj.weight1*x')';

            y3 = (obj.weight2*x' + obj.biases)';

            y = [y1 y2 y3];


        end

        function V = getHPs(obj)

            V = [obj.weight1(:);obj.weight2(:);obj.biases(:)];
        end

        function obj = setHPs(obj,V)

            nT1 = numel(obj.weight1(:));
            nT2 = numel(obj.weight2(:));

            obj.weight1 = reshape(V(1:nT1),size(obj.weight1));
            obj.weight2 = reshape(V(nT1+1:nT1+nT2),size(obj.weight2));

            obj.biases = V(nT1+nT2+1:end);
            
        end
    end
end