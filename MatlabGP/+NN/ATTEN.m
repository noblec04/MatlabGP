classdef ATTEN

    properties
        Keyweights
        Queryweights
        Valueweights
    end

    methods

        function obj = ATTEN(in,out)
            obj.Keyweights = normrnd(zeros(out,in),sqrt(1/in));
            obj.Queryweights = normrnd(zeros(out,in),sqrt(1/in));
            obj.Valueweights = normrnd(zeros(out,in),sqrt(1/in));
        end

        function [y] = forward(obj,x)

            k = obj.Keyweights*x';
            q = obj.Queryweights*x';
            v = obj.Valueweights*x';

            A = utils.softmax(q*k'/sqrt(size(k,1)));

            y = (A.*v)';

        end

        function V = getHPs(obj)

            V = [obj.Keyweights(:);obj.Queryweights(:);obj.Valueweights(:)];

        end

        function obj = setHPs(obj,V)

            nK = numel(obj.Keyweights(:));
            nQ = numel(obj.Queryweights(:));
            nV = numel(obj.Valueweights(:));

            obj.Keyweights = reshape(V(1:nK),size(obj.Keyweights));
            obj.Queryweights = reshape(V(nK+1:nK+nQ),size(obj.Queryweights));
            obj.Valueweights = reshape(V(nK+nQ+1:nK+nQ+nV),size(obj.Valueweights));
            
        end
    end
end