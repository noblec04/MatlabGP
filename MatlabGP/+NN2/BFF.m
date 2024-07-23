classdef BFF

    properties
        weight_mu
        weight_sig
        biases_mu
        biases_sig

        out
        in
    end

    methods

        function obj = BFF(in,out)
            obj.weight_mu = normrnd(zeros(out,in),sqrt(1/in));
            obj.weight_sig = 0.001*log(abs(normrnd(zeros(out,in),sqrt(1/in))));
            obj.biases_mu = normrnd(zeros(out,1),sqrt(1/in));
            obj.biases_sig = 0.001*log(abs(normrnd(zeros(out,1),sqrt(1/in))));

            obj.out = out;
            obj.in = in;
        end

        function [y] = forward(obj,x)
            
            epsilon_matrix = normrnd(zeros(obj.out,obj.in),ones(obj.out,obj.in));
            epsilon_vec = normrnd(zeros(obj.out,1),ones(obj.out,1));

            weight = obj.weight_mu + epsilon_matrix.*exp(obj.weight_sig);
            biases = obj.biases_mu + epsilon_vec.*exp(obj.biases_sig);

            
            y = (weight*x' + biases)';

        end

        function V = getHPs(obj)

            V = [obj.weight_mu(:);obj.weight_sig(:);obj.biases_mu;obj.biases_sig];
        end

        function obj = setHPs(obj,V)

            n1 = numel(obj.weight_mu(:));
            n2 = numel(obj.weight_sig(:));
            n3 = numel(obj.biases_mu(:));

            obj.weight_mu = reshape(V(1:n1),size(obj.weight_mu));
            obj.weight_sig = reshape(V(n1+1:n1+n2),size(obj.weight_sig));

            obj.biases_mu = reshape(V(n1+n2+1:n1+n2+n3),size(obj.biases_mu));
            obj.biases_sig = reshape(V(n1+n2+n3+1:end),size(obj.biases_sig));
            
        end
    end
end